import numpy as np
import matplotlib.pyplot as plt

def error_rate(Nx, error):
    """Compute the error rate."""

    # Compute the logs of the nodes and the errors
    logNx = np.log(np.array(Nx))
    logError = np.log(np.array(error))
    ones = np.ones(len(logNx))

    V = np.array([ones, logNx]).transpose()

    # Solve least squares system
    A = np.matmul(V.transpose(), V)
    b = np.matmul(V.transpose(), logError)

    c = np.linalg.solve(A, b)

    return c[1]


if __name__ == '__main__':

    #At t_final = 0.5
    #nodes = [8, 16, 32, 64, 128, 256, 512, 1024]
    #error_norms = [0.1435990269362068, 0.053295467449838894, 
                   #0.014159196232448148, 0.00201267238707246, 
                   #0.0002557494168653626, 3.2064690696880085e-05, 
                   #4.012308772348105e-06, 5.016219140951846e-07]

    #At t_final = 5.0
    nodes = [8, 16, 32, 64, 128, 256, 512, 1024]
    error_norms = [0.1435990269362068, 0.053295467449838894, 
                    0.02225572452093035, 0.0034820913238763672, 
                    0.00042836433759898976, 5.347270287171297e-05, 
                    6.646325302443257e-06, 8.305990053787972e-07]

    p = error_rate(nodes, error_norms)

    print('Error rate = {:6f} '.format(p))

    plt.loglog(nodes, error_norms, 'r-o')
    plt.xlabel('log(Nx)')
    plt.ylabel('log(error_norms)')
    plt.grid()
    plt.show()