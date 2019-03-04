class NewsItem:
    """
    class containing information about a news item
    """
    def __init__(self, identifier, content="",
                 dct=None):
        self.identifier = identifier  # string, the original document name in the dataset
        self.dct = dct                # e.g. "2005-05-14T02:00:00.000+02:00" -> document creation time
        self.content = content        # the text of the news article
        self.entity_mentions = []  # set of instances of EntityMention class
        
class EntityMention:
    """
    class containing information about an entity mention
    """

    def __init__(self, mention,
                 begin_index, end_index,
                 gold_link=None,
                 the_type=None, sentence=None, agdistis_link=None,
                 spotlight_link=None, aida_link='NIL'): #, exact_match=False):
        self.sentence = sentence         # e.g. 4 -> which sentence is the entity mentioned in
        self.mention = mention           # e.g. "John Smith" -> the mention of an entity as found in text
        self.the_type = the_type         # e.g. "Person" | "http://dbpedia.org/ontology/Person"
        self.begin_index = begin_index   # e.g. 15 -> begin offset
        self.end_index = end_index       # e.g. 25 -> end offset
        self.gold_link = gold_link       # gold link if existing
        self.agdistis_link = agdistis_link    # AGDISTIS link
        self.aida_link = aida_link             # AIDA link
        self.spotlight_link = spotlight_link             # Spotlight link
