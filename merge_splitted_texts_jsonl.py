import sys
import jsonlines

# Merges JSON lines if text was too long and therefore divided

if len(sys.argv)!=2:
    print('provide path to JSONL')

path_jsonlines=sys.argv[1]
      
data=[]

with jsonlines.open(path_jsonlines) as reader:
    for obj in reader:
      id=obj['id']
      if data and id==data[-1]['id']:
          last_obj=data[-1]
          last_obj_length=last_obj['length']
          for ner in obj['ner']:
            ner['start_pos']+=last_obj_length
            ner['end_pos']+=last_obj_length
            last_obj['ner'].append(ner)
          last_obj['length']+=obj['length']
      else:
          data.append(obj)

with jsonlines.open(path_jsonlines+'.merged', mode='w', compact=True) as writer:
    for obj in data:
        try:
            del obj['length']
        except KeyError:
            pass
        writer.write(obj)
   
