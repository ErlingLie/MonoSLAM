import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.linalg import block_diag
from visualization import draw_model_and_query_pose
import matplotlib.animation as animation
from matplotlib import pyplot as plt


def computeRootSIFTDescriptors(descriptor):
    descriptor /= np.linalg.norm(descriptor, ord=1, axis=1, keepdims=True)
    descriptor = np.sqrt(descriptor)
    return descriptor


def findKeypoints(image, mask=None, N=15):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(image, mask)
    return kp, des

def match2d(kp1, des1, kp2, des2, N):
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1,des2,k=2)
    # matches = bf.match(des1, des2)
    # Apply ratio test
    matches = sorted(matches, key = lambda x:x[0].distance)

    matchLocations = []
    descriptors = []
    good = []
    for m, n in matches[:N]:
        if m.distance < 0.75*n.distance:
            good.append(m)
            img1Pt = kp1[m.queryIdx].pt
            img2Pt = kp2[m.trainIdx].pt
            descriptors.append(des1[m.queryIdx])
            matchLocations.append([img1Pt[0],img1Pt[1], img2Pt[0], img2Pt[1]])
    return np.array(matchLocations), np.array(descriptors),  good



def triangulate_points(frame1, frame2, K):
    kp, des = findKeypoints(frame1, None, 3000)
    kp2, des2 = findKeypoints(frame2, None, 3000)
    matches, descriptors, good = match2d(kp,des, kp2, des2, 30)
    uv1 = matches[:,:2]
    uv2 = matches[:,2:4]
    kp = [kp[m.queryIdx] for m in good]
    E, innliers =cv2.findEssentialMat(uv1, uv2, K, method=cv2.RANSAC, prob=0.99, threshold=4)
    num_innliers, R, t, mask, X = cv2.recoverPose(E, uv1, uv2, K, distanceThresh=50)
    X = X[:,innliers.squeeze().astype(bool)]
    X/=X[-1,]
    X*=-1
    return X, kp, descriptors[innliers.squeeze().astype(bool), :], R, -t








def rightQuaternionMatrix(q):
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], q[3], -q[2]],
        [q[2], -q[3], q[0], q[1]],
        [q[3], q[2], -q[1], q[0]]
    ])
        
def leftQuaternionMatrix(q):
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], -q[3], q[2]],
        [q[2], q[3], q[0], -q[1]],
        [q[3], -q[2], q[1], q[0]]
    ])

def dqdphi(phi: np.ndarray):
    if not phi.any():
        return np.vstack([[0,0,0], np.eye(3)])
    phinorm = np.sqrt(np.sum(phi**2))
    dq0 = -np.sin(phinorm/2)*1/(2*phinorm)*phi.reshape([1,3])
    dq = np.sin(phinorm/2)*1/phinorm*np.eye(3) + \
        (1/2*np.cos(phinorm/2)/phinorm - np.sin(phinorm/2)/phinorm**2)/phinorm*phi.reshape([1,3])@phi.reshape([3,1])
    return np.block([[dq0],[dq]])


def skewSymmetric(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
        ])

def priorUpdate(X: np.ndarray,P:np.ndarray, Q: np.ndarray, dt):
    r = X[0:3]
    q = X[3:7]
    v = X[7:10]
    w = X[10:13]
    rp = r + v*dt
    dq = Rot.from_rotvec(w*dt)
    qp = Rot.from_quat([q[1],q[2],q[3],q[0]])*dq
    qp_wrong_order = qp.as_quat()
    qp = np.zeros(4)
    qp[0] = qp_wrong_order[3]
    qp[1:] = qp_wrong_order[:3]
    
    dq_quat_t = dq.as_quat()
    dq_quat = np.zeros(4)
    dq_quat[0] = dq_quat_t[3]
    dq_quat[1:] = dq_quat_t[:3]

    dr = np.block([[np.eye(3), np.zeros((3,3)), np.eye(3)*dt, np.zeros((3,3))]])
    dtheta = np.block([[np.zeros((3,3)), np.eye(3)-dt*skewSymmetric(w),
             np.zeros((3,3)), dt*np.eye(3)]])
    dv = np.block([[np.zeros((3,6)), np.eye(3), np.zeros((3,3))]])
    dw = np.block([[np.zeros((3,9)), np.eye(3)]])


    Fx = np.vstack([dr,dtheta,dv,dw])

    dru = np.block([[np.eye(3)*dt, np.zeros((3,3))]])
    dthetau = np.block([[np.zeros((3,3)), np.eye(3)*dt]])
    dvu = np.block([[np.eye(3), np.zeros((3,3))]])
    dwu = np.block([[np.zeros((3,3)), np.eye(3)]])

    Fu = np.vstack([dru, dthetau, dvu, dwu])
    P1 = P[:12,:12]
    Pp = Fx@P1@Fx.T + Fu@Q@Fu.T
    P[:12,:12] = Pp
    X[0:3] = rp
    X[3:7] = qp
    return X, P





def measurementModel(X: np.ndarray, P : np.ndarray, K: np.ndarray, h=512, w=512):
    x = X[0:3]
    q = X[3:7]
    M_W = X[13:].reshape(-1,3)
    R_WC = Rot.from_quat([q[1],q[2],q[3],q[0]])
    M_C = R_WC.apply(M_W)+x.T
    M_xy = K@M_C.T
    M_xy /= M_xy[2,:]
    inframe = M_xy[0,:] > 0
    inframe = np.logical_and(inframe, M_xy[0,:] < w)
    inframe = np.logical_and(inframe, M_xy[1,:] > 0)
    inframe = np.logical_and(inframe, M_xy[1,:] < h)
    inframe_idx = np.nonzero(inframe.squeeze())[0]
    

    return M_xy[:2,:], inframe_idx

def diffQuatRot(q,r):
    '''
    Given p' = q * p * q^T
    Calcululate dp'/dq
    '''
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    x = r[0]
    y = r[1]
    z = r[2]
    dq = np.array([
        [2*q0*x - 2*q3*y + 2*q2*z, 2*q1*x + 2*q2*y + 2*q3*z, 2*q1*y - 2*q2*x + 2*q0*z, 2*q1*z - 2*q0*y - 2*q3*x],
        [2*q3*x + 2*q0*y - 2*q1*z, 2*q2*x - 2*q1*y - 2*q0*z, 2*q1*x + 2*q2*y + 2*q3*z, 2*q0*x - 2*q3*y + 2*q2*z],
        [2*q1*y - 2*q2*x + 2*q0*z, 2*q3*x + 2*q0*y - 2*q1*z, 2*q3*y - 2*q0*x - 2*q2*z, 2*q1*x + 2*q2*y + 2*q3*z],
        ])
    return dq


def dh(X, idx, K):
    x = X[0:3]
    q = X[3:7]
    m = X[13+3*idx:16+3*idx]
    Ml = leftQuaternionMatrix(q)
    Mr = rightQuaternionMatrix(np.hstack([q[0],-q[1:]]))
    M = Ml@Mr
    M = M[1:,1:]
    dm = K@M
    dx = K
    X_C = K@(M@m+x)
    dh_dX_C = np.array([[1/X_C[2], 0, -X_C[0]/X_C[2]**2],
                        [0, 1/X_C[2], -X_C[1]/X_C[2]**2]])

    dh_dq = dh_dX_C@K@diffQuatRot(q,m-x)

    Hx = np.zeros([2,X.shape[0]])
    Hx[:,:3] = dh_dX_C@dx
    Hx[:,3:7] = dh_dq
    Hx[:,13+3*idx:16+3*idx] = dh_dX_C@dm


    Q_dtheta = 1/2*np.vstack([-q[None,1:],q[0]*np.eye(3)+skewSymmetric(q[1:])])
    X_dx = block_diag(np.eye(3), Q_dtheta, np.eye(6), np.eye(X.shape[0]-13) )
    H = Hx@X_dx
    return H


def ellipse(x,y,x0,y0,alpha,sx,sy):
    return ((np.cos(alpha)*(x-x0) - np.sin(alpha)*(y-y0))/sx)**2 + ((np.sin(alpha)*(x-x0) + np.cos(alpha)*(y-y0))/sy)**2 

def getSearchMask(X, P, K, h, w, R):
    xc, _ = measurementModel(X,P,K,h,w)
    N = xc.shape[1]
    H = np.vstack([dh(X,i,K) for i in range(N)])
    Sigma = H@P@H.T + np.kron(np.eye(N), R)
    mask = np.zeros([h,w, 3],dtype= np.uint8)
    ellipses = []
    for j in range(N):
        p = Sigma[2*j:2+2*j, 2*j:2+2*j]
        [e, v] = np.linalg.eig(p)
        alpha = np.arctan(v[1,0]/v[0,0])

        x0 = xc[0,j]
        y0 = xc[1,j]
        try:
            ellipses.append({'x0': x0, 'y0': y0, 'alpha': alpha, 'sx': e[0], 'sy': e[1]})
            cv2.ellipse(mask, [int(x0), int(y0)], [int(e[0]/10), int(e[1]/10)], alpha, 0, 360, (255, 0, 0),-1)
        except:
            mask[:] = 1
    return mask[:,:,0], ellipses
        

def drawEllipses(image, ellipses):
    for ellipse in ellipses:
        x0 = ellipse['x0']
        y0 = ellipse['y0']
        sx = ellipse['sx']
        sy = ellipse['sy']
        alpha = ellipse['alpha']
        try:
            cv2.ellipse(image, [int(x0), int(y0)], [int(sx/10), int(sy/10)], alpha, 0, 360, (0, 0, 255),3)
        except:
            pass

def measurementUpdate(z, idx, X: np.ndarray, P: np.ndarray, K: np.ndarray, R):
    z_est = measurementModel(X,P,K)[0][:, idx].T
    dz = z-z_est
    error_meassurements = np.where(np.linalg.norm(dz,2,1)>50)[0]
    idx = np.delete(idx, error_meassurements)
    dz = np.delete(dz, error_meassurements, 0)

    H = np.vstack([dh(X,i,K) for i in idx])

    Sigma_IN = H@P@H.T + np.kron(np.eye(len(idx)),R)
    K_kalman = P@H.T@np.linalg.inv(Sigma_IN)
    dx = K_kalman@dz.reshape(-1,1)
    M = (np.eye(P.shape[0])-K_kalman@H)
    P = M@P@M.T + K_kalman@np.kron(np.eye(len(idx)),R)@K_kalman.T
    
    dx = dx.squeeze()
    X[0:3] += dx[0:3]
    qm = (Rot.from_quat([X[4], X[5], X[6], X[3]])*Rot.from_rotvec(dx[3:6])).as_quat()
    X[4:7] = qm[0:3]
    X[3] = qm[3]
    X[7:] += dx[6:]

    # Covariance reset
    G = np.eye(3)-skewSymmetric(dx[3:6]/2)
    P[3:6, 3:6] = G@P[3:6,3:6]@G.T 
    return X.squeeze(), P



if __name__ == "__main__":
    filename = r"C:\Users\erling\Pictures\Camera Roll\test.mp4"
    cap = cv2.VideoCapture(filename)
    # Read until video is completed
    ret, baseFrame = cap.read()

    h = baseFrame.shape[0]
    w = baseFrame.shape[1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    K = np.array([[1000, 0 ,1280/2], [0, 1000, 720/2], [0,0,1]])
    ret, frame2 = cap.read()
    R_noise = 6*np.eye(2)
    Q = 2*np.eye(6)*1/16
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))
    M, markers, descriptors, R, t = triangulate_points(baseFrame, frame2, K)
    N = M.shape[1]
    X = np.zeros((13+N*3))
    q = Rot.from_matrix(R).as_quat()
    X[3] =  1# q[3]
    # X[4:7] = q[:3]
    # X[:3] = t.squeeze()
    for i in range(N):
        X[13+3*i:16+3*i] = M[:3,i].squeeze()
    

    # P is now one dimension smaller as we represent the orientation error in 3 dimensions
    P = np.eye(X.shape[0]-1)*1e-8
    # P[6:,6:]*= 1000

    mask1 = np.full([h,w],255,np.uint8)
    print(N)

    bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    image = np.zeros([h,2*w,3])

    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            X, P = priorUpdate(X, P, Q, 1/30)
            print(X[:3])
            print(X[13:16])
            mask, ellipses = getSearchMask(X,P,K,h,w, R_noise)

            kps, des =  findKeypoints(frame, mask)
            matchMask = np.array([[ellipse(kps[i].pt[0], kps[i].pt[1], **ellipses[j]) <= 0.01 for j in range(N)] for i in range(len(kps))], dtype=np.uint8)
            matches = bfMatcher.match(des, descriptors, matchMask)
            matches = sorted(matches, key=lambda x: (x.trainIdx, x.distance))
            i = -1
            bestMatches = []
            for m in matches:
                if m.trainIdx != i:
                    i = m.trainIdx
                    bestMatches.append(m)

            idx = np.array([m.trainIdx for m in bestMatches])
            z = np.array([kps[m.queryIdx].pt for m in bestMatches])
            X, P = measurementUpdate(z, idx,X, P, K, R_noise)

            # frame = cv2.drawKeypoints(frame,kps,frame)

            image = cv2.drawMatches(frame,kps,baseFrame,markers,bestMatches, image)
            uncertainties = drawEllipses(image, ellipses)
            cv2.imshow( "Frame", cv2.resize(image,(w,h//2)) )
            frames.append([X[13:].reshape(-1,3).T,\
                np.block([[Rot.from_quat([X[4],X[5],X[6],X[3]]).as_matrix(),\
                     X[:3,None]], [0,0,0,1]]), K])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    


    cap.release()
    cv2.destroyAllWindows()
    fig = plt.figure()

    lookfrom = np.array([[0,0,-15]])
    ani = animation.FuncAnimation(fig, lambda x: draw_model_and_query_pose(*x,\
         point_size=50,
         lookfrom=lookfrom, lookat=X[0:3]+Rot.from_quat([X[4],X[5],X[6],X[3]]).apply([0,0,1])*2,
        #  lookfrom=np.linalg.inv(x[1])[:3,-1]+x[1][2,:3]*10, lookat=np.array([-3.11418413e-02, -1.27069648e+01,  7.78613041e+01]),
         ), frames, interval = 30)
    plt.show()

