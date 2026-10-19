import hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
class ReleaseRegressions(unittest.TestCase):
    def test_resume_rolls_back_uncommitted_validation_and_best_weights(self):
        self.assertIsNotNone(importlib.util.find_spec('checkpoint_support'),'Recovery transaction helper missing')
        from checkpoint_support import recovery_snapshot,recover_history
        from artifact_support import verify_validation_selection
        with tempfile.TemporaryDirectory() as td:
            work=Path(td)
            committed={'event':'validation','step':5000,'count':2,'raw_loss':10.,'ema_loss':11.,'raw_diagnostics':[],'ema_diagnostics':[]}
            (work/'metrics.jsonl').write_text(json.dumps(committed)+'\n')
            best={'model':{'w':torch.tensor([10.])},'step':5000,'best_metric':10.,'weight_variant':'raw'}
            torch.save(best,work/'best.pth');state=recovery_snapshot(work)
            (work/'metrics.jsonl').write_bytes((work/'metrics.jsonl').read_bytes()+json.dumps(dict(committed,step=8000,raw_loss=5.)).encode()+b'\n{"partial":')
            torch.save(dict(best,step=8000,best_metric=5.),work/'best.pth')
            recover_history(work,state)
            rows=[json.loads(line) for line in (work/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1)
            saved=torch.load(work/'best.pth',weights_only=True)
            self.assertEqual(saved['step'],5000)
            rows.append({'event':'finished','step':5000})
            status={'state':'training_finished','step':5000,'best_metric':10.,'best_variant':'raw'}
            self.assertEqual(verify_validation_selection(rows,status,saved,2)['loss'],10.)
            self.assertTrue(any((work/'recovery_history').glob('*.jsonl')))
            (work/'metrics.jsonl').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'history'):recover_history(work,state)
    def test_evaluation_identity_distinguishes_cpu_and_cuda(self):
        import model_runtime
        self.assertTrue(hasattr(model_runtime,'backend_identity'),'Backend identity is missing from evaluation')
        cpu=model_runtime.backend_identity('cpu');gpu=model_runtime.backend_identity('cuda:0')
        self.assertEqual(cpu['device_type'],'cpu');self.assertEqual(gpu['device_type'],'cuda')
        self.assertNotEqual(json.dumps(cpu,sort_keys=True),json.dumps(gpu,sort_keys=True))
    def test_evaluation_identity_tracks_network_configuration_and_sources(self):
        import model_runtime
        self.assertTrue(hasattr(model_runtime,'network_identity'),'Effective network identity missing')
        with tempfile.TemporaryDirectory() as td:
            bundle=Path(td)
            cfg=bundle/'configs/order2.json';cfg.parent.mkdir()
            code=bundle/'third_party/score_sde/sde_2nd/models/ncsnpp.py';code.parent.mkdir(parents=True)
            cfg.write_text('{"model":{"scale_by_sigma":true}}');code.write_text('original architecture')
            original=model_runtime.network_identity(bundle)
            cfg.write_text('{"model":{"scale_by_sigma":false}}')
            changed_config=model_runtime.network_identity(bundle)
            self.assertNotEqual(original,changed_config)
            code.write_text('changed architecture')
            self.assertNotEqual(changed_config,model_runtime.network_identity(bundle))
    def test_paper_iteration_redraw_explains_nonfinite_curves(self):
        import plot_reference_results as plot
        self.assertTrue(hasattr(plot,'draw_iterations'),'Iteration annotation helper missing')
        rows=plot.read_csv(ROOT/'results/reference/fresh_paper_iterations.csv')
        fig=plot.draw_iterations(rows,'all_trials','paper')
        labels=' '.join(t.get_text() for t in fig.axes[0].texts)
        self.assertIn('7/3000',labels)
        for kappa in (0,5,10):self.assertIn(f'kappa={kappa}:',labels)
        plot.plt.close(fig)
if __name__=='__main__':unittest.main()
