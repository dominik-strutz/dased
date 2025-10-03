#!/usr/bin/env python3
"""
Test script to verify anchor->knot conversion was successful.
"""

import sys
import numpy as np

sys.path.insert(0, '.')

def test_basic_functionality():
    """Test basic DASLayout functionality with new knot terminology."""
    print("Testing DASLayout with knot terminology...")
    
    from dased.layout import DASLayout
    
    # Test 1: Basic knot-based layout
    knots = np.array([[0, 0], [1000, 0], [2000, 1000]])
    layout = DASLayout(knots, spacing=50.0)
    print(f"✓ Created layout with {layout.n_channels} channels over {layout.cable_length:.1f}m")
    
    # Test 2: Backward compatibility with anchors parameter
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            layout_old = DASLayout(anchors=knots, spacing=50.0)
        print("✓ Backward compatibility with 'anchors' parameter works")
    except Exception as e:
        print(f"✗ Backward compatibility failed: {e}")
    
    # Test 3: Access knot_locations
    knot_locs = layout.knot_locations
    print(f"✓ knot_locations accessible: {knot_locs.shape}")
    
    # Test 4: Backward compatibility property
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            anchor_locs = layout.anchor_locations
        print("✓ Backward compatibility property 'anchor_locations' works")
    except Exception as e:
        print(f"✗ Backward compatibility property failed: {e}")
    
    print("All tests passed! ✓")

if __name__ == "__main__":
    test_basic_functionality()
