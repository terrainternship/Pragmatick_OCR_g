import votemodules
from votemodules.voteparser import vparser
from votemodules.votemodel import Resnext50ml, Resnet50s


#doc_path = "../images/1-71-1-color.jpg"
#doc_path = "../images/1-71-2-color.jpg"
doc_path = "../images/1-71-2-vote.jpg"

jpath='result'
device='cuda'
device='cpu'
vmodel = Resnext50ml('weights/chpt35-col.pth',device, 3)
smodel = Resnet50s('weights/vote_resnet50_sign.pt',device, 2)
vp = vparser(vmodel, smodel)
vp.process_data(doc_path,0.5)
vp.save_json_resut(jpath)
