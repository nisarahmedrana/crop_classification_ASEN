import numpy as np
import pandas as pd

def compute_vegetation_indices(df):
    """
    Expects a DataFrame `df` with the following columns:
    Band1, Band2, Band3, Band4, Band5, Band6, Band7, Band8, Band9, Band10, Band11
    """

    eps = 1e-10  # Small epsilon to avoid division by zero

    # NDVI: (Band 5 - Band 4) / (Band 5 + Band 4)
    df['NDVI'] = (df['Band5'] - df['Band4']) / (df['Band5'] + df['Band4'] + eps)

    # EVI: 2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)
    df['EVI'] = 2.5 * (df['Band5'] - df['Band4']) / (df['Band5'] + 6 * df['Band4'] - 7.5 * df['Band2'] + 1 + eps)

    # SAVI: ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    L = 0.5
    df['SAVI'] = ((df['Band5'] - df['Band4']) / (df['Band5'] + df['Band4'] + L + eps)) * (1 + L)

    # GNDVI: (NIR - Green) / (NIR + Green)
    df['GNDVI'] = (df['Band5'] - df['Band3']) / (df['Band5'] + df['Band3'] + eps)

    # RENDVI: (Band5 - Band4) / (Band5 + Band4)
    df['RENDVI'] = (df['Band5'] - df['Band4']) / (df['Band5'] + df['Band4'] + eps)

    # MSAVI: 0.5 * (2 * NIR + 1 - sqrt((2*NIR + 1)^2 - 8 * (NIR - Red)))
    df['MSAVI'] = 0.5 * (2 * df['Band5'] + 1 - np.sqrt((2 * df['Band5'] + 1)**2 - 8 * (df['Band5'] - df['Band4']) + eps))

    # NDWI: (Green - NIR) / (Green + NIR)
    df['NDWI'] = (df['Band3'] - df['Band5']) / (df['Band3'] + df['Band5'] + eps)

    # NDRE: (Band5 - Band4) / (Band5 + Band4)
    df['NDRE'] = (df['Band5'] - df['Band4']) / (df['Band5'] + df['Band4'] + eps)

    # SR: NIR / Red
    df['SR'] = df['Band5'] / (df['Band4'] + eps)

    # PRI: (Red - NIR) / (Red + NIR)
    df['PRI'] = (df['Band4'] - df['Band5']) / (df['Band4'] + df['Band5'] + eps)

    return df
