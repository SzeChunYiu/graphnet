# hibeam_det.py
# Detector mapping that zeroes time and passes through spatial coords + signal_bkg.

from graphnet.models.detector.detector import Detector


class HIBEAM_Detector(Detector):
    """Custom detector that:
    - keeps dom_x, dom_y, dom_z as-is,
    - zeroes dom_t (we don't want timing),
    - passes signal_bkg through as a node feature.
    """

    # Geometry / id fields (keep whatever your pipeline expects)
    xyz = ["x", "y", "z"]
    string_id_column = "string"
    string_index_name = "string"
    sensor_id_column = "sensor_id"

    # Helper: create a zero tensor matching the shape/dtype/device of x
    def _zeros(self, x):
        return x.new_zeros(x.shape)

    def feature_map(self):
        """Map pulsemaps column names -> tensor transforms."""
        return {
            "dom_x": self._identity,
            "dom_y": self._identity,
            "dom_z": self._identity,
            "dom_t": self._zeros,       # keep t but zero it out
            "signal_bkg": self._identity,  # NEW: pass-through the feature (0/1 or float)
        }
