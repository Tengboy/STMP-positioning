import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).absolute().parents[1]/'scripts'))
from train_support import BatchStream, freeze_teacher, optimizer_update

class TrainingContracts(unittest.TestCase):
    def test_resume_preserves_next_batch_and_rng(self):
        data=torch.arange(7).reshape(7,1).float()
        stream=BatchStream(data,3,42)
        stream.next(); stream.next()
        state=stream.state_dict()
        resumed=BatchStream(data,3,999,state=state)
        for _ in range(4): self.assertTrue(torch.equal(stream.next(),resumed.next()))

    def test_teacher_is_frozen_while_student_changes(self):
        teacher=torch.nn.Linear(1,1,bias=False)
        student=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad(): teacher.weight.fill_(2); student.weight.fill_(0)
        freeze_teacher(teacher)
        opt=torch.optim.SGD(student.parameters(),lr=0.1)
        x=torch.tensor([[1.]])
        loss=optimizer_update(student,opt,lambda: (student(x)-teacher(x)).square().mean(),1,0.1,0)
        self.assertTrue(torch.equal(teacher.weight,torch.tensor([[2.]])))
        self.assertIsNone(teacher.weight.grad)
        self.assertGreater(student.weight.item(),0)
        self.assertEqual(loss,4.)

    def test_nonfinite_loss_does_not_update_weights(self):
        model=torch.nn.Linear(1,1,bias=False)
        before=model.weight.detach().clone()
        opt=torch.optim.SGD(model.parameters(),lr=0.1)
        with self.assertRaises(FloatingPointError):
            optimizer_update(model,opt,lambda: model.weight.sum()*float('nan'),1,0.1,0)
        self.assertTrue(torch.equal(model.weight,before))

if __name__=='__main__': unittest.main()
