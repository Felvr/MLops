"""Additional crash consistency, tamper and identity checks on corrected code."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import lab
import release


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'w';lab.initialize(self.root)
        self.m=lab.read(self.root/'manifest-B.json');self.r=lab.read(self.root/'report-B.json')

    def test_crash_between_pointer_and_journal_recovers_once(self):
        with self.assertRaises(RuntimeError): release.publish_event(self.root,self.m,self.r,True)
        self.assertTrue((self.root/'pending.json').exists())
        self.assertEqual(lab.read(self.root/'accepted.json')['model_id'],'fixture-B')
        for _ in range(2):
            r=subprocess.run([sys.executable,str(Path(lab.__file__)),'--workspace',str(self.root),'run'],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(len((self.root/'publications.jsonl').read_text().splitlines()),1)
        self.assertFalse((self.root/'pending.json').exists())

    def test_tampered_artifact_rejected_before_publish(self):
        before=(self.root/'accepted.json').read_bytes()
        (self.root/'model-B.txt').write_text('CORRUPTED')
        with self.assertRaises(ValueError): release.publish_event(self.root,self.m,self.r)
        self.assertEqual(before,(self.root/'accepted.json').read_bytes())

    def test_release_identity_binds_runtime_and_decision(self):
        old=release.release_id(self.m,self.r)
        r=copy.deepcopy(self.r);r['evaluation_run_id']='new-independent-evaluation'
        self.assertNotEqual(old,release.release_id(self.m,r))
        m=copy.deepcopy(self.m);m['runtime']['generation']['max_new_tokens']=16
        m['runtime_sha256']=release.digest(release.canonical(m['runtime']))
        self.assertNotEqual(old,release.release_id(m,r))

    def test_error_before_replace_preserves_pointer(self):
        before=(self.root/'accepted.json').read_bytes()
        with patch('os.fsync',side_effect=OSError('Injected fsync error')):
            with self.assertRaises(OSError): release.publish(self.m,self.r,self.root/'accepted.json')
        self.assertEqual(before,(self.root/'accepted.json').read_bytes())
        self.assertFalse(list(self.root.glob('*.tmp')))
