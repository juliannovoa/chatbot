import logging
import random
import re
from collections import defaultdict
from enum import Enum
from typing import Any, Mapping, Callable

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from src.knowledge_graph import KnowledgeGraph
from src.multimedia import Multimedia
from src.ner import NameEntityRecognitionModelBERT


class QuestionType(Enum):
    MULTIMEDIA = 1
    RECOMMENDATION = 2
    ONE_ENTITY = 3
    TWO_ENTITIES = 4
    UNK = -1
    WHO_OF = '(.*)who(\'is|\'re| is| are| was| were) (the )?(((?! of ).)*) of (the movie )?(.*)'
    WH_OF = '(.*)(what|which|where|when)(\'is|\'re| is| are| was| were) (the )?(((?! of ).)*) of (the movie )?(.*)'


class QuestionSolver:
    EXCUSES = ("Sorry, I didn't get that, could you reformulate your question?",
               "I am sorry. I did not understand your question. Could you say it in a different way?",
               "I am not sure if I can help you. Could you ask the question with other words?",
               "I am still learning. Maybe I can help you if you ask in a different manner :)")
    RECOMMENDATION_KEYWORDS = ("recommend", "recomend", "suggest", "sugest", "similar")
    MULTIMEDIA_KEYWORDS = ("pictur", "imag", "poster", "frame")

    def __init__(self):
        self._ner_model = NameEntityRecognitionModelBERT()
        self._knowledge_graph = KnowledgeGraph()
        self._multimedia = Multimedia()
        self._stop_words = set(stopwords.words('english'))
        self._questions: Mapping[QuestionType, Callable[[str], str]] = {
            QuestionType.MULTIMEDIA: self._process_multimedia,
            QuestionType.RECOMMENDATION: self._process_recommendation,
            QuestionType.WHO_OF: self._process_who_of_question,
            QuestionType.WH_OF: self._process_wh_of_generic_question,
            QuestionType.ONE_ENTITY: self._process_one_entity_question,
            QuestionType.TWO_ENTITIES: self._process_two_entities_question
        }

    @classmethod
    def _generate_excuse(cls) -> str:
        return random.choice(cls.EXCUSES)

    def answer_question(self, question: str) -> str:
        question_type = self._get_question_type(question)
        try:
            return self._questions[question_type](question)
        except ValueError:
            return self._generate_excuse()

    def _get_question_type(self, question: str) -> QuestionType:
        question = question.rstrip('?')
        # Check if it is a recommendation or multimedia question
        words = word_tokenize(question)
        stemmer = PorterStemmer()
        for word in words:
            stemmed_word = stemmer.stem(word)
            if stemmed_word in self.RECOMMENDATION_KEYWORDS:
                return QuestionType.RECOMMENDATION
            if stemmed_word in self.MULTIMEDIA_KEYWORDS:
                return QuestionType.MULTIMEDIA

        # Classify the factual question
        n_entities = len(self._ner_model.find_name_entities(question))
        if n_entities == 1:
            return QuestionType.ONE_ENTITY
        if n_entities == 2:
            return QuestionType.TWO_ENTITIES
        if re.match(QuestionType.WHO_OF.value, question):
            return QuestionType.WHO_OF
        if re.match(QuestionType.WH_OF.value, question):
            return QuestionType.WH_OF
        return QuestionType.UNK

    def _remove_stop_words(self, sentence: str) -> str:
        word_tokens = word_tokenize(sentence)
        filtered_sentence = [w for w in word_tokens if not w.lower() in self._stop_words]
        return ' '.join(filtered_sentence)

    def _process_multimedia(self, question: str) -> str:
        return self._multimedia.process_question(question)

    def _process_recommendation(self, question: str) -> str:
        return question

    def _process_one_entity_question(self, question: str) -> str:
        logging.debug(f'Processing one entity question')
        question = question.rstrip('?')
        ner_result = self._ner_model.find_name_entities(question)
        named_entities = list(ner_result.values())[0]
        for named_entity in named_entities:
            question = re.sub(named_entity, '', question)
        clean_question = self._remove_stop_words(question)

        entities = self._knowledge_graph.get_closest_node(" ".join(named_entities), predicate=False)
        predicates = self._knowledge_graph.get_closest_node(clean_question, predicate=True)

        logging.debug(f'Entities: {entities}')
        logging.debug(f'Predicates: {predicates}')

        logging.debug(f'Search crowd information')

        if crowd_question := self._knowledge_graph.get_crowd_information_object(predicates, entities):
            return self._process_crowd_question(crowd_question)

        logging.debug(f'Crowd information not found')
        query = '''
            PREFIX ddis: <http://ddis.ch/atai/> 
            PREFIX wd: <http://www.wikidata.org/entity/> 
            PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
            PREFIX schema: <http://schema.org/> 
            SELECT DISTINCT ?x WHERE {{ 
                <{e}> <{p}> ?x .
            }}
        '''
        if information := self._process_query_search(query, predicates, entities):
            return self.process_response(information)

        # TODO: usar embeddings.

        raise ValueError('No answer for this question')

    def _process_two_entities_question(self, question: str) -> str:
        question = question.rstrip('?')
        ner_result = self._ner_model.find_name_entities(question)
        entities = list(ner_result.values())[0]
        for entity in entities:
            question = re.sub(entity, '', question)
        question = self._remove_stop_words(question)
        predicates = self._knowledge_graph.get_closest_node(question, predicate=True)
        entities = defaultdict(list)
        for idx, item in enumerate(items):
            for entity in self._knowledge_graph.get_closest_node(item):
                entities[idx].append(entity)

        logging.debug(f'Processing two entity question')
        logging.debug(f'Entities: {entities}')
        logging.debug(f'Predicates: {predicates}')

        # logging.debug(f'Search crowd information')
        # crowd_information = self._knowledge_graph.get_crowd_information_object(predicates, entities)
        # if len(crowd_information) != 0:
        #     return self.process_crowd_information(crowd_information)
        # logging.debug(f'Crowd information not found')
        query = '''
            PREFIX ddis: <http://ddis.ch/atai/> 
            PREFIX wd: <http://www.wikidata.org/entity/> 
            PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
            PREFIX schema: <http://schema.org/> 
            SELECT DISTINCT ?x WHERE {{ 
                <{e1}> <{p}> ?x .
            }}
        '''
        if information := self._process_query_search(query, predicates, entities):
            return self.process_response(information)
        return self._generate_excuse()

    def _process_wh_of_generic_question(self, question: str) -> str:
        question = question.lower().rstrip('?')
        match = re.match(QuestionType.WH_OF.value, question)
        if match is None:
            raise ValueError('Invalid question')
        predicates = self._knowledge_graph.get_closest_node(match.group(5), predicate=True)
        entities = self._knowledge_graph.get_closest_node(match.group(8))
        logging.debug(f'Entities: {entities}')
        logging.debug(f'Predicates: {predicates}')
        logging.debug(f'Search crowd information')
        crowd_information = self._knowledge_graph.get_crowd_information_object(predicates, entities)
        if len(crowd_information) != 0:
            return self._process_crowd_question(crowd_information)
        logging.debug(f'Crowd information not found')
        query = '''
            PREFIX ddis: <http://ddis.ch/atai/> 
            PREFIX wd: <http://www.wikidata.org/entity/> 
            PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
            PREFIX schema: <http://schema.org/> 
            SELECT DISTINCT ?x WHERE {{ 
                <{e}> <{p}> ?x .
            }}
        '''
        information = self._process_query_search(query, predicates, entities)

        if len(information) == 0:
            logging.debug('Information not found. Trying second degree search')
            query = '''
                PREFIX ddis: <http://ddis.ch/atai/> 
                PREFIX wd: <http://www.wikidata.org/entity/> 
                PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
                PREFIX schema: <http://schema.org/> 
                SELECT DISTINCT ?x ?y WHERE {{
                    <{e}> ?a ?y . 
                    ?y <{p}> ?x .
                }}
            '''
            information = self._process_query_search(query, predicates, entities, True)
        if len(information) == 0:
            return self._generate_excuse()
        else:
            return self.process_response(information)

    def _process_who_of_question(self, question: str) -> str:
        question = question.lower().rstrip('?')
        match = re.match(QuestionType.WHO_OF.value, question)
        if match is None:
            raise ValueError('Invalid question')
        predicates = self._knowledge_graph.get_closest_node(match.group(4), predicate=True)
        entities = self._knowledge_graph.get_closest_node(match.group(7))
        logging.debug(f'Entities: {entities}')
        logging.debug(f'Predicates: {predicates}')
        logging.debug(f'Search crowd information')
        crowd_information = self._knowledge_graph.get_crowd_information_object(predicates, entities)
        if len(crowd_information) != 0:
            return self._process_crowd_question(crowd_information)
        logging.debug(f'Crowd information not found')
        query = '''
            PREFIX ddis: <http://ddis.ch/atai/> 
            PREFIX wd: <http://www.wikidata.org/entity/> 
            PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
            PREFIX schema: <http://schema.org/> 
            SELECT DISTINCT ?x WHERE {{ 
                <{e}> <{p}> ?x .
                ?x wdt:P31 wd:Q5. 
            }}
        '''
        information = self._process_query_search(query, predicates, entities)

        if len(information) == 0:
            logging.debug('Information not found. Trying second degree search')
            query = '''
                PREFIX ddis: <http://ddis.ch/atai/> 
                PREFIX wd: <http://www.wikidata.org/entity/> 
                PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
                PREFIX schema: <http://schema.org/> 
                SELECT DISTINCT ?x ?y WHERE {{
                    <{e}> ?a ?y . 
                    ?y <{p}> ?x .
                    ?x wdt:P31 wd:Q5.  
                }}
            '''
            information = self._process_query_search(query, predicates, entities, True)
        if len(information) == 0:
            return self._generate_excuse()
        else:
            return self.process_response(information)

    def _process_query_search(self, query: str, predicates: list, entities: list, early_stop=False) -> defaultdict[set]:
        information = defaultdict(set)
        logging.debug('Start query.')
        logging.debug(f'Entities: {entities}')
        logging.debug(f'Predicates: {predicates}')
        for predicate in predicates:
            for entity in entities:
                logging.debug(f'Query elements: {predicate}(P:{type(predicate)}) and {entity}(E:{type(entity)})')
                formatted_query = query.format(e=entity, p=predicate)
                for row in self._knowledge_graph.query(formatted_query):
                    obj = row.x.toPython()
                    try:
                        subj = row.y.toPython()
                    except AttributeError:
                        subj = entity
                    information[(subj, predicate)].add(obj)
                if len(information) >= 1 and early_stop:
                    return information

        return information

    def process_query_check(self, query: str, predicates: list, entities_1: list, entities_2: list,
                            early_stop=True) -> bool:
        information = defaultdict(set)
        logging.debug('Start query.')
        logging.debug(f'Entities 1: {entities_1}')
        logging.debug(f'Entities 2: {entities_2}')
        logging.debug(f'Predicates: {predicates}')
        for predicate in predicates:
            for entity_1 in entities_1:
                for entity_2 in entities_2:
                    formatted_query = query.format(e=entity, p=predicate)
                for row in self._knowledge_graph.query(formatted_query):
                    obj = row.x.toPython()
                    try:
                        subj = row.y.toPython()
                    except AttributeError:
                        subj = entity
                    information[(subj, predicate)].add(obj)
                if len(information) >= 1 and early_stop:
                    return information

        return information

    def process_response(self, information: defaultdict) -> str:
        output = []
        for (s, p), o in information.items():
            predicate = self._knowledge_graph.get_node_label(p, is_predicate=True)
            subj = self._knowledge_graph.get_node_label(s)
            description = self.get_film_description(s)
            if len(o) == 1:
                for it in o:
                    if self._knowledge_graph.element_is_entity(it):
                        obj = self._knowledge_graph.get_node_label(it)
                    else:
                        obj = it
                    output.append(f'{obj} is the {predicate} of {subj}{description}')
            else:
                output.append(f'The {predicate}s of {subj}{description} are:')
                for it in o:
                    obj = self._knowledge_graph.get_node_description(it)
                    output.append(f'   {obj}')

        return '\n'.join(output)

    def get_film_description(self, film: str) -> str:
        query = f'''
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    SELECT DISTINCT ?x WHERE {{
                        <{film}> wdt:P577 ?x .
                    }}
                '''
        logging.debug('Start description query.')

        for row in self._knowledge_graph.query(query):
            date = row.x.toPython()
            if isinstance(date, int):
                return f'({date})'
            else:
                return f'({date.year})'

        query = f'''
                    PREFIX ns2: <http://schema.org/>
                    SELECT DISTINCT ?x WHERE {{
                        <{film}> ns2:description ?x . 
                    }}
                '''
        for row in self._knowledge_graph.query(query):
            return f'({row.x.toPython()})'

        return ''

    def _process_crowd_question(self, crowd_question: Mapping[str, Any]) -> str:
        subject = self._knowledge_graph.get_node_label(crowd_question['subject'], is_predicate=False, short_name=True)
        predicate = self._knowledge_graph.get_node_label(crowd_question['predicate'], is_predicate=True,
                                                         short_name=True)
        obj = self._knowledge_graph.get_node_label(crowd_question['object'], is_predicate=False, short_name=True)

        n_correct = crowd_question['n_correct']
        n_incorrect = crowd_question['n_incorrect']
        output = [
            f'{obj} is the {predicate} of {subject}.',
            f'The crowd had an inter-rate agreement of {crowd_question["kappa"]} in this batch.'
        ]
        if n_incorrect > n_correct:
            output.append(f'The crowd thinks this answer is incorrect '
                          '({n_correct} votes for correct and {n_incorrect} votes for incorrect).')
            if crowd_question['mistake_position'] == 'Predicate':
                output.append(f'Someone has said that the wrong information is {predicate}.')
                if crowd_question['right_label']:
                    new_pred = self._knowledge_graph.get_node_label(
                        crowd_question['right_label'], is_predicate=True, short_name=True)
                    output.append(f'They think that {obj} is the {new_pred} of {subject}.')
            elif crowd_question['mistake_position'] == 'Object':
                output.append(f'Someone has said that the wrong information is {obj}.')
                if crowd_question['right_label']:
                    new_obj = self._knowledge_graph.get_node_label(
                        crowd_question['right_label'], is_predicate=False, short_name=True)
                    output.append(f'They think that {new_obj} is the {predicate} of {subject}.')
            elif crowd_question['mistake_position'] == 'Subject':
                output.append(f'Someone has said that the wrong information is {subject}.')
                if crowd_question['right_label']:
                    new_subj = self._knowledge_graph.get_node_label(
                        crowd_question['right_label'], is_predicate=False, short_name=True)
                    output.append(f'They think that {obj} is the {predicate} of {new_subj}.')
        elif n_incorrect <= n_correct:
            output.append(f'The crowd thinks this answer is correct '
                          '({n_correct} votes for correct and {n_incorrect} votes for incorrect).')
        return '\n'.join(output)
