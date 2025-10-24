import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from dwarforge.utils import get_coord_median, get_tile_numbers


def test_get_coord_median_raises_on_empty():
    """Tests that an empty list raises a ValueError."""
    with pytest.raises(ValueError, match='Input list cannot be empty.'):
        get_coord_median([])


def test_get_coord_median_single_coordinate_returns_same():
    """Tests that a single coordinate returns itself as the median."""
    c = SkyCoord(ra=123.4, dec=-5.6, frame='icrs', unit='deg')
    m = get_coord_median([c])
    assert np.isclose(m.ra.deg, 123.4)  # type: ignore
    assert np.isclose(m.dec.deg, -5.6)  # type: ignore


def test_get_coord_median_no_ra_wrap():
    """Tests median calculation without RA wrapping."""
    coords = [
        SkyCoord(ra=10.0, dec=0.0, frame='icrs', unit='deg'),
        SkyCoord(ra=20.0, dec=2.0, frame='icrs', unit='deg'),
        SkyCoord(ra=30.0, dec=4.0, frame='icrs', unit='deg'),
    ]
    median = get_coord_median(coords)
    assert np.isclose(median.ra.deg, 20.0)  # type: ignore
    assert np.isclose(median.dec.deg, 2.0)  # type: ignore


def test_get_coord_median_with_ra_wrap_close_to_zero():
    """Tests that coordinates near the RA=0/360 boundary are handled correctly."""
    coords = [
        SkyCoord(ra=359.0, dec=6.0, frame='icrs', unit='deg'),
        SkyCoord(ra=1.0, dec=12.0, frame='icrs', unit='deg'),
        SkyCoord(ra=0.5, dec=35.0, frame='icrs', unit='deg'),
    ]
    median = get_coord_median(coords)
    assert np.isclose(median.ra.deg, 0.5)  # type: ignore
    assert np.isclose(median.dec.deg, 12.0)  # type: ignore


def test_ra_no_wrap_when_range_small():
    # Range < 180°, so no special handling needed
    coords = [
        SkyCoord(ra=170, dec=-1.0, frame='icrs', unit='deg'),
        SkyCoord(ra=175, dec=0.0, frame='icrs', unit='deg'),
        SkyCoord(ra=190, dec=1.0, frame='icrs', unit='deg'),
    ]
    m = get_coord_median(coords)
    assert np.isclose(m.ra.deg, 175.0)  # type: ignore
    assert np.isclose(m.dec.deg, 0.0)  # type: ignore


def test_get_coord_median_with_ra_wrap_at_large_angles():
    """Tests that RA ranges > 180 degreees are handled correctly."""
    coords = [
        SkyCoord(ra=10.0, dec=-10.0, frame='icrs', unit='deg'),
        SkyCoord(ra=200.0, dec=0.0, frame='icrs', unit='deg'),
        SkyCoord(ra=210.0, dec=10.0, frame='icrs', unit='deg'),
    ]
    m = get_coord_median(coords)
    assert np.isclose(m.ra.deg, 210.0)  # type: ignore
    assert np.isclose(m.dec.deg, 0.0)  # type: ignore


def test_get_tile_numbers_calexp():
    """Test get_tile_numbers for files with 'calexp' in the name."""
    filename = 'calexp-CFIS_234_295.fits'
    tile_numbers = get_tile_numbers(filename)
    assert tile_numbers == (234, 295)


@pytest.mark.parametrize(
    'filename, tile_numbers',
    [
        ('CFIS_LSB.234.295.r.fits', (234, 295)),
        ('CFIS.234.295.u.fits', (234, 295)),
        ('PSS.DR4.234.295.i.fits', (234, 295)),
        ('PS-DR3.234.295.i.fits', (234, 295)),
        ('WISHES.234.295.z.fits', (234, 295)),
    ],
)
def test_get_tile_numbers_no_calexp(filename, tile_numbers):
    """Test get_tile_numbers for files without 'calexp' in the name."""
    assert get_tile_numbers(filename) == tile_numbers


@pytest.mark.parametrize(
    'filename, tile_numbers',
    [
        ('CFIS_LSB.003.017.r.fits', (3, 17)),
        ('CFIS.020.024.u.fits', (20, 24)),
        ('PSS.DR4.011.005.i.fits', (11, 5)),
        ('PS-DR3.014.001.i.fits', (14, 1)),
    ],
)
def test_get_tile_numbers_with_zfill(filename, tile_numbers):
    """Test get_tile_numbers for zero-padded bands."""
    assert get_tile_numbers(filename) == tile_numbers


@pytest.mark.parametrize(
    'filename, tile_numbers',
    [
        ('WISHES.9.99.z.fits', (9, 99)),
        ('calexp-CFIS_7_12.fits', (7, 12)),
    ],
)
def test_get_tile_numbers_without_zfill(filename, tile_numbers):
    """Test get_tile_numbers for non-zero-padded bands."""
    assert get_tile_numbers(filename) == tile_numbers
