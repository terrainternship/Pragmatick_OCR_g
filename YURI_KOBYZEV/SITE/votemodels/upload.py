import gradio as gr
import subprocess
import json
import os


class Upload:
    def __init__(self): 
        self.create_ui()
        pass

    def create_ui(self):
        def dirjson(sel): 
            print('dirjson:',sel)
            path=[]
            for j in sel:
                print('json:',j)
                f = open(j)
                data = json.load(f)
                print('data:',data)
                path.append(data)
            return path

 
        def dirimages(selected,predict): 
            if predict:
                sel = ' '.join(selected)
                print('python ./vote_parser.py --source ',sel)
                subprocess.call('python ./vote_parser.py --source '+sel,shell=True)
            result = gr.FileExplorer(root='./result', height=200, interactive=True, label="json available", file_count="multiple", glob="**/*.json")
            return selected,result

        def uploadtmp(dir): 
            newlist=[]
            listfiles = [d.name for d in dir]
            for f in listfiles:
                fname='images/'+os.path.basename(f)
                os.rename(f,fname)
                newlist.append(fname)
            return newlist,newlist
#            return [d.name for d in dir]

        with gr.Blocks() as demo:
            # load images from client to tmp dir
            input = gr.File(label="image files",file_count="multiple",file_types=["image"],)
            files = gr.Textbox()
            # move images from tmp dir to ./images
            show = gr.Button(value="Переместить временные загрузки")
            show.click(uploadtmp, input, [files])
             
            # preview images in gallery
            predict = gr.Checkbox(value=True,label="Предикт")
            showdir2gallery = gr.Button(value="Переместить выбрааные в галерею")
            images = gr.FileExplorer(root='./images', height=200, interactive=True, label="images available", file_count="multiple", glob="**/*.jpg")
            gallery = gr.Gallery(label="", show_label=False, elem_id="gallery", columns=[3], rows=[1], object_fit="contain", height="auto") 

            # preview selected json files
            result = gr.FileExplorer(root='./result', height=200, interactive=True, label="json available", file_count="multiple", glob="**/*.json")
            jsondir2gallery = gr.Button(value="Preview result json files")
            jsongallery = gr.JSON()
            jsondir2gallery.click(dirjson,[result],[jsongallery])
            showdir2gallery.click(dirimages,[images,predict],[gallery,result])

