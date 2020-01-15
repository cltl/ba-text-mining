import urllib
from rdflib import Graph, URIRef
from lab3_classes import NewsItem, EntityMention

def normalizeURL(s):
    """
    Normalize a URI by removing its Wikipedia/DBpedia prefix.
    """
    if s:
        if s.startswith('http://aksw.org/notInWiki'):
            return 'NIL'
        else:
            return urllib.parse.unquote(s.replace("http://en.wikipedia.org/wiki/", "").
                                        replace("http://dbpedia.org/resource/", ""). 
                                        replace("http://dbpedia.org/page/", "").
                                        strip().
                                        strip('"'))
    else:
        return 'NIL'

def load_article_from_nif_file(nif_file):
    """
    Load a dataset in NIF format.
    """
    g=Graph()
    g.parse(nif_file, format="n3")

    news_items=[]

    articles = g.query(
    """ SELECT ?articleid ?date ?string
    WHERE {
        ?articleid nif:isString ?string .
        OPTIONAL { ?articleid <http://purl.org/dc/elements/1.1/date> ?date . }
    }
    """)
    for article in articles:
        news_item_obj=NewsItem(
            content=article['string'],
            identifier=article['articleid'].strip(), 
            dct=article['date']
        )
        query=""" SELECT ?id ?mention ?start ?end ?gold
        WHERE {
            ?id nif:anchorOf ?mention ;
            nif:beginIndex ?start ;
            nif:endIndex ?end ;
            nif:referenceContext <%s> .
            OPTIONAL { ?id itsrdf:taIdentRef ?gold . }
        } ORDER BY ?start""" % str(article['articleid'])
        qres_entities = g.query(query)
        for entity in qres_entities:
            gold_link=normalizeURL(str(entity['gold']))
            if gold_link.startswith('http://aksw.org/notInWiki'):
                gold_link='NIL'
            entity_obj = EntityMention(
                begin_index=int(entity['start']),
                end_index=int(entity['end']),
                mention=str(entity['mention']),
                gold_link=gold_link
            )
            news_item_obj.entity_mentions.append(entity_obj)
        news_items.append(news_item_obj)
    return news_items
