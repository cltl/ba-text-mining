# import the default library for regular expressions matching, called `re`
import re
from nltk import Tree
import requests

def get_entities_of_type(a_type, a_doc):
    """
    Get all entities of type a_type in a_doc.
    """
    return filter(lambda w: w.ent_type_ == a_type, a_doc)

def find_closest_entity(entities, prop_position, e_type):
    """
    Find entities of a certain type and with the smallest distance to a property value.
    """
    min_distance=9999
    closest_entity=None
    for ent in entities:
        if ent.label_!=e_type and e_type is not None: 
            continue # skip entities of different types
            
        # determine the distance between the entity and the property
        distance=abs(ent.start_char-prop_position)
        if min_distance>distance and distance>0:
            min_distance=distance
            closest_entity=ent
    if closest_entity:
        return closest_entity.text
    else:
        return None
    
def check_for_pattern(doc, i, pattern):
    """
    Check a specific pattern appears just before the property value with token index "i". 
    Returns True or False.
    """
    pattern_tokens=pattern.split(' ')
    num_tokens=len(pattern_tokens)
    if num_tokens>=i: # if there are not enough tokens for the pattern to 'fit' before i
        return False
    tokens=[]
    for x in reversed(range(1, num_tokens+1)):
        prev_index=i-x
        tokens.append(doc[prev_index].text.lower())
        
    return tokens==pattern_tokens

def pattern_found_on_the_left(doc, token_index, patterns):
    """
    Check whether any of the patterns appears just before the property value with token index "token_index". 
    Returns True or False.
    """
    for pattern in patterns:
        if check_for_pattern(doc, token_index, pattern)==True:
            return True
    return False
        
def extract_year_from_date(date):
    """
    Extract the year value from a date by looking for four consecutive digits.
    """
    match = re.findall('\d{4}', date)
    first_match=match[0]
    return int(first_match)    

def evaluate_property(sys_property_data, gold_property_data):
    """
    Compare the system output to the gold data to compute precision, recall, and F1-score. 
    """
    tp=0
    fp=0
    fn=0
    for entity, gold_value in gold_property_data.items():
        if entity in sys_property_data and sys_property_data[entity]: 
            system_value=sys_property_data[entity]
            if system_value==gold_value:
                tp+=1
            else:
                fp+=1
                fn+=1
        else:
            fn+=1
        
    if tp+fp>0:
        precision=tp/(tp+fp)
    else:
        precision=0
    if tp+fn>0:
        recall=tp/(tp+fn)
    else:
        recall=0
    if precision+recall>0:
        f1=2*precision*recall/(precision+recall)
    else:
        f1=0
    
    return precision, recall, f1

def tok_format(tok):
    return "_".join([tok.orth_, tok.tag_, tok.dep_])


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)
    
def obtain_results_from_api(url, params):
    try:
        r=requests.get(url, params=params)
        print(r.request.url)
    except Exception as e:
        print('Error with wikipage', url, params, e)
        return {}
    j=r.json()
    if 'batchcomplete' not in j.keys() and 'parse' not in j.keys():
        print(r.request.url)
    return j
    
def get_wikipedia_page(title, lang='en'):
    params_extracts={
        'format': 'json',
        'action': 'query',
        'prop': 'extracts',
        'explaintext': True,
        'titles': title,
        'redirects': True,
        'exlimit': 1
    }
    url='https://%s.wikipedia.org/w/api.php?' % lang

    j=obtain_results_from_api(url, params_extracts)
    data={}
    for page_id, page_info in j['query']['pages'].items():
        if page_id=='-1': continue
        data['title']=page_info['title']
        data['extract']=page_info['extract']
    return data