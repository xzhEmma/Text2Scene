# import os
# path = '/root/Text2Scene/data/graphs_txt1'
# files= os.listdir(path)
# graphs_objs = []
# for k in range(10019): #遍历文件夹     
#         with open(path+"/"+"graph_%05d.txt"%(k), 'rb') as f:
#                 #print(path+"/"+"graph_%05d.txt"%(k))
#                 graph = f.read()
#                 graph = eval(graph)
#                 graph_objs_ls = graph['entities']
#                 graph_objs = []
#                 for i in range (len(graph_objs_ls)):   
#                         graph_obj = graph_objs_ls[i]['span']
                        
#                         graph_objs.append(graph_obj)# ['Mike', 'a hat', 'horns', 'Mike', 'a ball', 'Jenny', 'An owl', 'the tree'] 
                
                        
                        
#         graphs_objs.append(graph_objs)

# objects = []
# for k in range(len(graphs_objs)):
#         objs = graphs_objs[k]
#         for cur_objs in objs:    
#                 cur_objs = cur_objs.lower()
#                 cur_objs = further_token_process(cur_objs)
#                 print(cur_objs)
#                 cur_objs   = [self.lang_vocab.word2index[w] for w in cur_objs]
#         print(objs)
#         objects.append(cur_objs)

                        
                

# import sng_parser
# from collections import defaultdict
# import json
# import copy
# from nltk.corpus import wordnet as wn
# import numpy as np
# import json
# import os



# def txt_strtonum_feed(filename):
#     sentences=[]
#     with open(filename, 'r') as f:
#         line = f.readline()
#         while line:
#             eachline = line.replace('\n','').split('\t')###按行读取文本文件，每行数据以列表形式返回
#             sentences.append(eachline)
#             line = f.readline()
#             line = f.readline()
#         return sentences

# def save_txt_results(text, count, text_dir):
#     if not os.path.exists(text_dir):
#                 os.makedirs(text_dir)
#     f = open('%s/graph_%05d.txt'%(text_dir, count), 'a')
#     f.write(json.dumps(text))
#     f.close()




# def build_graph(path):
#     with open(path, 'rb') as f:
#         desc = f.read()
#         desc = eval(desc)
#         desc_entities = desc['entities']
#         desc_relations = desc['relations']
#         # count the number of nodes
#         count_nodes = len(desc_entities)
#         adj = np.zeros((count_nodes, count_nodes))
#         # handle adjacent
#         for desc_relation in desc_relations:
#             sub = desc_relation['subject']
#             obj = desc_relation['object']
#             # rel = desc_relation['relation']
#             adj[sub][obj] = 1
#     print(adj)
#     return adj

# # def obj_extra(obj)
# #
# #     input_inds, input_lens = encode_sentences(obj)
# #     out = TextEncoder(input_inds, input_lens)
# #     return out

# def main():
#     scene_graph_dict('/root/Text2Scene/data/AbstractScenes_v1.1/SimpleSentences/SimpleSentences2_10020.txt','/root/Text2Scene/data/graphs_txt2')
#     #build_graph('/tmp/text2scene/data/AbstractScenes_v1.1/graph_txt/count_000001.txt')

# if __name__ == '__main__':
#     main()
