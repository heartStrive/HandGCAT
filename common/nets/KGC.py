import os
import sys
sys.path.append(os.path.abspath('./common/nets'))
from graph_utils import *

def get_graph_L():
    mano = MANO()
    joint_hori_conn = ((1,5), (5,9), (9,13), (13,17), (2,6),(6,10),(10,14), (14,18), (3,7), (7,11), (11,15), (15,19),(4,8),(8,12),(12,16),(16,20))
    graph_Adj, graph_L = build_graphs(joint_num=21,skeleton=mano.skeleton,flip_pairs=joint_hori_conn)
    return graph_L

class KnowledgeGuidedModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.graph_L = get_graph_L()
        
        self.CL_K = [1, 1, 1, 1]
        self.CL_F = [in_channel, 1024, 2048, 2048, out_channel]
        _cl = []
        _bn = []
        for i in range(len(self.CL_K)):
            Fin = self.CL_K[i] * self.CL_F[i]
            Fout = self.CL_F[i+1]

            _cl.append(nn.Linear(Fin, Fout))
            scale = np.sqrt(2.0 / (Fin + Fout))
            _cl[-1].weight.data.uniform_(-scale, scale)
            _cl[-1].bias.data.fill_(0.0)

            _bn.append(nn.BatchNorm1d(Fout))

        self.cl = nn.ModuleList(_cl)
        self.bn = nn.ModuleList(_bn)
    
    def forward(self, kypt_feats):
        for i in range(len(self.CL_K)):
            kypt_feats = graph_conv_cheby(kypt_feats,self.cl[i],self.bn[i],self.graph_L,self.CL_F[i+1],self.CL_K[i])
        
        return kypt_feats