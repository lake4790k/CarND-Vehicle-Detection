
class FeaturesConfig:

    def __init__(self):
        self.color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

        self.hog_feat = True  # HOG features on or off
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

        self.spatial_feat = True  # Spatial features on or off
        self.spatial_size = (16, 16)  # Spatial binning dimensions

        self.hist_feat = True  # Histogram features on or off
        self.hist_bins = 16  # Number of histogram bins

        self.y_start = 380
        self.y_stop = 650
