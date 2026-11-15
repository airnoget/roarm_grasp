from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
from PIL import Image
from dataset_labels import save_labels
from prepare_targets import export, records, split_records


class ExportTests(unittest.TestCase):
    def test_scene_and_duplicate_images_never_cross_split(self):
        rows = [dict(id=str(i), group=str(i), sha256=str(i), stage='single', labels=[[0,.5,.5,.2,.2]]) for i in range(20)]
        rows[1]['group'] = rows[0]['group']
        rows[2]['sha256'] = rows[1]['sha256']
        first = split_records(rows)
        self.assertEqual(first, split_records(rows))
        a, b = first['train'], first['val']
        for key in ('group', 'sha256'):
            self.assertFalse({r[key] for r in a} & {r[key] for r in b})
        self.assertEqual(len(a)+len(b),20)

    def test_export_validates_labels_and_preserves_sources(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.test-artifacts') as temp:
            base=Path(temp); session=base/'raw'
            for i in range(10):
                folder=session/str(i);folder.mkdir(parents=True)
                Image.new('RGB',(40,30),(i*20,40,50)).save(folder/'original_rgb.png')
                (folder/'capture.json').write_text(json.dumps(dict(annotation_image='original_rgb.png',
                    annotation_status='unlabelled',split_group=str(i),collection_stage='single')))
                save_labels(folder,[[3,4,20,25]],[40,30])
            source=(session/'0/capture.json').read_bytes()
            m=export(session,base/'export')
            self.assertEqual([len(m['splits'][k]) for k in ('train','val')],[8,2])
            self.assertEqual((session/'0/capture.json').read_bytes(),source)
            self.assertEqual(len(list((base/'export/labels').glob('*/*.txt'))),10)
            import hashlib
            profile=session/'camera-preset.json';profile.write_text('{"preset":"fixture"}')
            digest=hashlib.sha256(profile.read_bytes()).hexdigest()
            for i in range(10):
                path=session/str(i)/'capture.json';meta=json.loads(path.read_text())
                meta['camera_profile']=dict(preset_file=profile.name,preset_sha256=digest)
                path.write_text(json.dumps(meta))
            prepared=export(session,base/'export-profile')
            self.assertEqual(prepared['camera_preset_sha256'],digest)
            profile.write_text('{"preset":"changed"}')
            with self.assertRaisesRegex(ValueError,'Preset snapshot changed'):records(session)
            profile.write_text('{"preset":"fixture"}')
            (session/'0/original_rgb.txt').write_text('0 .1 .1 .1 .1\n')
            with self.assertRaisesRegex(ValueError,'disagrees'):records(session)

    def test_unreviewed_capture_is_not_background(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.test-artifacts') as temp:
            folder=Path(temp)/'0';folder.mkdir()
            (folder/'capture.json').write_text('{"annotation_status":"unlabelled"}')
            with self.assertRaisesRegex(ValueError,'Unreviewed'):records(temp)


if __name__=='__main__':unittest.main()
