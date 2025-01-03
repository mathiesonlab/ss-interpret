import allel
import numpy as np

def predict_ihs_max(matrix: np.ndarray) -> float:
    """Computes ihs statistic values from genetic data and then returns the maximum abs value of the statistics

    Parameters
    ----------
    data List[np.ndarray]: [genetic data, genetic positions]

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    genetic_data = matrix[:,:,0].transpose()
    intersnp = matrix[:,:,1][0] # all the same
    positions = [sum(intersnp[:i]) for i in range(len(intersnp))]
    assert len(pos) == len(intersnp)

    #if not isinstance(data, list):
    #    raise Exception('The ihs test statistic has multiple inputs')
    #genetic_data = data[0][:, :, 0]
    #positions = data[1][:]
    #haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int)
    h1 = allel.HaplotypeArray(haplos)
    ihs = allel.ihs(h1, pos=positions, include_edges=True)
    output = float(np.nanmax(np.abs(ihs))) # TODO do we need this line?
    return output