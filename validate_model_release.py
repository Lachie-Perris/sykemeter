"""Verify that forecast, matrices and explorer all use the same model release."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def validate(skip_forecast=False):
    config=json.loads((ROOT/'model_config.json').read_text())
    assert config==json.loads((ROOT/'docs/model_config.json').read_text())
    for name,digest in config['matrix_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,name
    import generate_site
    import surf_forecast as forecast
    generate_site.validate_required_runtime_files()
    assert forecast.SPOT_LAT==config['spot']['latitude']
    assert forecast.SPOT_LON==config['spot']['longitude']
    assert forecast.GFS_GRID_LAT==round(forecast.SPOT_LAT*4)/4
    assert forecast.GFS_GRID_LON==round(forecast.SPOT_LON*4)/4
    matrices=forecast.load_tide_transfer_matrices(forecast.TIDE_TRANSFER_MATRIX_PATHS)
    # Check the actual runtime conversion at low, mid, high and interpolated tide.
    probe=pd.DataFrame(dict(wave_height_m=[2.0]*4,wave_period_s=[10.0]*4,
                            wave_direction_deg=[90.0]*4,tide_height_m=[0,1.5,3,2.25]))
    transformed=forecast.apply_tide_aware_transfer_matrices(probe,matrices)
    row,col=forecast.ROW_MAPPING[90],forecast.COL_MAPPING[10]
    expected=np.array([matrices[-1.5][row,col],matrices[0.0][row,col],matrices[1.5][row,col],
                       (matrices[0.0][row,col]+matrices[1.5][row,col])/2])
    np.testing.assert_allclose(transformed.nearshore_wave_height_m,2*expected)
    explorer=json.loads((ROOT/'docs/assets/swan_model_explorer_data.json').read_text())
    assert len(explorer['cases'])==90 and len(explorer['case_options'])==30
    assert explorer['metadata']['model_version']==config['model_version']
    spot=explorer['metadata']['spot']
    assert (spot['x_m'],spot['y_m'])==(config['spot']['swan_xi_m'],config['spot']['swan_eta_m'])
    assert (spot['row'],spot['col'])==(config['spot']['swan_output_row'],config['spot']['swan_output_col'])
    cases=pd.read_csv(ROOT/'docs/assets/swan_wave_cases.csv').set_index('case_id')
    assert len(cases)==90 and cases.final_accuracy_percent.ge(99.5).all()
    for case_id,case in explorer['cases'].items():
        assert case_id.startswith('cleaned_tp_'),case_id
        assert np.isclose(case['spot_hs_m'],cases.loc[case_id,'spot_Hs_m'],atol=5e-5)
        assert np.isclose(case['grids']['hsign_m'][spot['row']][spot['col']],case['spot_hs_m'],atol=5e-5)
        assert case['offshore_tp_s']<=17.071
    html=(ROOT/'docs/swan_model_explorer.html').read_text()
    assert '"mda_case_013"' not in html
    assert 'cleaned_tp_mda_001' in html
    assert '(spot.col + 0.5) * cellWidth' in html
    assert '(rows - spot.row - 0.5) * cellHeight' in html
    # Every relative URL referenced by the landing page must exist locally.
    index=(ROOT/'docs/index.html').read_text()
    for url in re.findall(r'(?:src|href)="([^"]+)"',index):
        if url.startswith(('https:','http:','mailto:','#')):
            continue
        assert (ROOT/'docs'/url.split('?')[0].split('#')[0]).exists(),url
    if not skip_forecast:
        svg=ET.parse(ROOT/'docs/final_surf_forecast.svg').getroot()
        metadata=next(e for e in svg.iter() if e.attrib.get('id')=='sykemeter-model')
        assert json.loads(metadata.text)==config,'Public SVG uses stale model configuration'
    print('PASS: runtime location, tide interpolation, matrix hashes, 90 explorer cases, spot marker and public assets.')
    if not skip_forecast:
        print('PASS: generated public forecast embeds this exact model configuration.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-forecast',action='store_true')
    validate(parser.parse_args().skip_forecast)
