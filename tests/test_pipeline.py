"""Boundary tests use tiny artificial fixtures, never training/benchmark inputs."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from build_market import feature_matrix, metrics
from verify_sources import require_verified

class Boundaries(unittest.TestCase):
    def test_future_changes_do_not_change_features(self):
        rng=np.random.default_rng(7)
        units=rng.poisson(.4,(5,500)).astype(float); orders=(units>0).astype(float)
        prices=np.ones(5); first=np.zeros(5); origin=400
        before=feature_matrix(units,origin,prices,first,orders,True)
        units[:,origin:]=999999; orders[:,origin:]=999999
        np.testing.assert_array_equal(before,feature_matrix(units,origin,prices,first,orders,True))

    def test_no_actual_sales_has_no_percentage_error(self):
        result=metrics(np.zeros(3),np.ones(3))
        self.assertIsNone(result['wape']); self.assertEqual(result['mae'],1)

    def test_unverified_bytes_are_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root=Path(tmp); (root/'transactions').mkdir()
            (root/'transactions/online_retail_II.xlsx').write_bytes(b'changed')
            with self.assertRaises(ValueError): require_verified(root,['retail'])

if __name__=='__main__': unittest.main()
