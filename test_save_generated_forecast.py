"""Regression tests for concurrent automated forecast commits."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from xml.sax.saxutils import escape

SCRIPT=Path(__file__).with_name('save_generated_forecast.py')


def git(cwd,*args):
    return subprocess.check_output(['git',*args],cwd=cwd,text=True,stderr=subprocess.DEVNULL).strip()


def forecast(repo,model,text,cycle='20261026-06Z'):
    (repo/'docs/final_surf_forecast.svg').write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><metadata id="sykemeter-model">'+
        escape(json.dumps(model))+'</metadata><text>'+text+'</text></svg>')
    (repo/'.forecast_state/latest_gfs_cycle.txt').write_text(cycle+'\n')


class SaveForecastTests(unittest.TestCase):
    def test_concurrent_svg_and_stale_release_guards(self):
        with tempfile.TemporaryDirectory(prefix='forecast-save-test-') as folder:
            root=Path(folder)
            remote=root/'origin.git'
            git(root,'init','--bare','--initial-branch=main',str(remote))
            a,b=root/'a',root/'b'
            git(root,'clone',str(remote),str(a))
            for name in ['docs','.forecast_state']:
                (a/name).mkdir()
            model={'model_version':'v1'}
            (a/'model_config.json').write_text(json.dumps(model))
            (a/'README.txt').write_text('initial')
            forecast(a,model,'base')
            git(a,'config','user.name','Test'); git(a,'config','user.email','test@example.invalid')
            git(a,'add','.'); git(a,'commit','-m','Initial'); git(a,'push','origin','main')
            git(root,'clone',str(remote),str(b))
            git(b,'config','user.name','Test'); git(b,'config','user.email','test@example.invalid')
            # First workflow advances main while the second generates a different complete SVG.
            forecast(a,model,'first workflow')
            (a/'README.txt').write_text('concurrent change must survive')
            git(a,'add','.'); git(a,'commit','-m','First workflow'); git(a,'push','origin','main')
            forecast(b,model,'second workflow')
            result=subprocess.run([sys.executable,str(SCRIPT)],cwd=b,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('second workflow',git(b,'show','origin/main:docs/final_surf_forecast.svg'))
            self.assertEqual((b/'README.txt').read_text(),'concurrent change must survive')
            # Refuse an older model generated while main advances to a new model.
            git(a,'pull','--ff-only','origin','main')
            (a/'model_config.json').write_text(json.dumps({'model_version':'v2'}))
            git(a,'add','.'); git(a,'commit','-m','New model'); git(a,'push','origin','main')
            forecast(b,model,'stale model result')
            result=subprocess.run([sys.executable,str(SCRIPT)],cwd=b,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('skipped stale forecast',result.stdout)
            self.assertNotIn('stale model result',(b/'docs/final_surf_forecast.svg').read_text())
            # Do not discard unrelated working-tree changes.
            (b/'README.txt').write_text('uncommitted work')
            result=subprocess.run([sys.executable,str(SCRIPT)],cwd=b,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual((b/'README.txt').read_text(),'uncommitted work')


if __name__=='__main__':
    unittest.main()
