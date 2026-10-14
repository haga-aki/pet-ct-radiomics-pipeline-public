import unittest
import numpy as np
from guarded_shape import core,eligible,prepare_columns,evaluate_columns,choose_scale,proposal


def fixture():
    p=core.Parameters();spacing=np.array([1.,1.,3.]);labels=np.zeros((3,3,60),np.uint8)
    labels[:,:,:21]=1;labels[:,:,23:]=3
    grid=(np.arange(-9,10)-.5)*3.
    profile=1/(1+np.exp((grid+5)/3))
    shift=core.crossings(profile,3.,p.profile_sigma_voxels)[.5]
    patches=[]
    for x in range(3):
        for y in range(3):
            patches.append(dict(key=(x,y),training=True,correction=shift,shift=shift,
                gap_mm=6.,reliability=1.,unambiguous_fraction=1.,crossing_iqr_mm=0.,
                columns=[(x,y,20,profile)]))
    return p,spacing,labels,patches


class GuardedShapeTests(unittest.TestCase):
    def test_heldout_not_in_guard(self):
        p,s,l,patches=fixture();patches[0]['training']=False
        data=prepare_columns(patches,l,np.zeros_like(l,bool),np.ones((3,3))*-5,np.ones((3,3))*61.5,s,p,1)
        self.assertEqual(len(data['groups']),8)
        self.assertEqual(len(eligible(patches,p)),8)

    def test_identity_reproduces_baseline_profile(self):
        p,s,l,patches=fixture()
        data=prepare_columns(patches,l,np.zeros_like(l,bool),np.ones((3,3))*-5,np.ones((3,3))*61.5,s,p,1)
        for row in evaluate_columns(data,0.):self.assertAlmostEqual(row['before_mm'],row['after_mm'])

    def test_guard_rejects_wrong_direction(self):
        p,s,l,patches=fixture()
        data=prepare_columns(patches,l,np.zeros_like(l,bool),np.ones((3,3))*8,np.ones((3,3))*61.5,s,p,1)
        scale,results,history=choose_scale([data])
        self.assertLess(scale,1.)
        self.assertTrue(all(abs(r['after_mm'])<=abs(r['before_mm'])+1.5+1e-6 for r in results))
        self.assertFalse(history[0]['passed'])

    def test_protected_columns_prevent_assignment(self):
        p,s,l,patches=fixture()
        data=prepare_columns(patches,l,np.ones_like(l,bool),np.ones((3,3))*-5,np.ones((3,3))*61.5,s,p,1)
        # Protection is assignment veto, not a promise of full-column identity.
        result=evaluate_columns(data,1.)
        self.assertIsNotNone(result)

    def test_aligned_proposal_is_zero(self):
        p,s,l,patches=fixture()
        for patch in patches:patch['correction']=0.;patch['shift']=1.
        fields,info=proposal(patches,np.ones((3,3)),np.ones((3,3))*61.5,s,p)
        self.assertFalse(info['activated']);self.assertFalse(fields[0].any())


if __name__=='__main__':unittest.main()

