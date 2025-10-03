import numpy as np

def get_angle_encoding_unitary_Z(data):

    U = np.array([[1.0]], dtype=complex)
    
    for i in range(len(data)):

        RZ = np.array([
            [np.exp(-1j * data[i] / 2), 0],
            [0, np.exp(1j * data[i] / 2)]
        ], dtype=complex)
        
        U = np.kron(U, RZ)
    
    return U

def angle_encoding_unitary_mixed(x):

    U = np.array([[1.+0j]])

    for theta in x:
        RZ = np.array([[np.exp(-1j*theta[0]/2), 0],
                       [0, np.exp(1j*theta[0]/2)]], dtype=complex)
        
        RX = np.array([[np.cos(theta[1]/2), -1j*np.sin(theta[1]/2)],
                       [-1j*np.sin(theta[1]/2), np.cos(theta[1]/2)]], dtype=complex)
        
        RY = np.array([[np.cos(theta[2]/2), -np.sin(theta[2]/2)],
                       [np.sin(theta[2]/2),  np.cos(theta[2]/2)]], dtype=complex)
        
        U1 = RZ @ RX @ RY
        U = np.kron(U, U1)
    return U


def angle_encoding_unitary_Z_X(x):

    U = np.array([[1.+0j]])

    for theta in x:
        RZ = np.array([[np.exp(-1j*theta[0]/2), 0],
                       [0, np.exp(1j*theta[0]/2)]], dtype=complex)
        
        RX = np.array([[np.cos(theta[1]/2), -1j*np.sin(theta[1]/2)],
                       [-1j*np.sin(theta[1]/2), np.cos(theta[1]/2)]], dtype=complex)
        
        U1 = RZ @ RX
        U = np.kron(U, U1)
    return U