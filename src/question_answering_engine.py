import logging
import random
import re
from collections import defaultdict
from dataclasses import astuple
from enum import Enum
from typing import Mapping, Callable, List

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from src import Fact
from src.crowdsourcing import CrowdQuestion
from src.knowledge_graph import KnowledgeGraph
from src.multimedia import Multimedia
from src.ner import NameEntityRecognitionModelBERT
from src.utils import remove_stop_words


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
        self._multimedia = Multimedia(self._knowledge_graph, self._ner_model)
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
        except Exception as e:
            logging.error(e)

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

    def _process_multimedia(self, question: str) -> str:
        return self._multimedia.process_question(question)

    def _process_recommendation(self, question: str) -> str:
        raise NotImplemented('Cannot do recommendations')

    def _process_one_entity_question(self, question: str) -> str:
        logging.debug(f'Processing one entity question')
        question = question.rstrip('?')
        ner_result = self._ner_model.find_name_entities(question)
        named_entities = list(ner_result.values())[0]
        for named_entity in named_entities:
            question = re.sub(named_entity, '', question)
        clean_question = remove_stop_words(self._stop_words, question)

        entities = self._knowledge_graph.find_closest_node(" ".join(named_entities), predicate=False)
        predicates = self._knowledge_graph.find_closest_node(clean_question, predicate=True)

        logging.debug(f'Search crowd questions')
        if crowd_question := self._knowledge_graph.find_crowd_question(predicates, entities):
            return self._process_crowd_question(crowd_question)
        logging.debug(f'Crowd questions not found')

        logging.debug(f'Search knowledge graph')
        if facts := self._knowledge_graph.find_facts(predicates, entities):
            return self._process_facts(facts)
        logging.debug(f'Knowledge graph facts not found')

        # TODO: use embeddings.

        raise ValueError('No answer for this question')

    def _process_two_entities_question(self, question: str) -> str:
        logging.debug(f'Processing two entity question')
        question = question.rstrip('?')
        ner_result = self._ner_model.find_name_entities(question)

        for named_entities in ner_result.values():
            for named_entity in named_entities:
                question = re.sub(named_entity, '', question)
        clean_question = remove_stop_words(self._stop_words, question)
        predicates = self._knowledge_graph.find_closest_node(clean_question, predicate=True)
        entities1 = self._knowledge_graph.find_closest_node(" ".join(list(ner_result.values())[0]), predicate=False)
        entities2 = self._knowledge_graph.find_closest_node(" ".join(list(ner_result.values())[1]), predicate=False)

        logging.debug(f'Processing two entity question')
        logging.debug(f'Entities1: {entities1}')
        logging.debug(f'Entities2: {entities2}')
        logging.debug(f'Predicates: {predicates}')

        logging.debug(f'Search crowd questions (two entities)')
        if crowd_question := self._knowledge_graph.find_crowd_question_two_entities(predicates, entities1, entities2):
            return self._process_crowd_question(crowd_question)
        logging.debug(f'Crowd questions not found')

        logging.debug(f'Search knowledge graph (two entities)')
        if facts := self._knowledge_graph.find_facts_two_entities(predicates, entities1, entities2):
            return self._process_facts(facts)
        logging.debug(f'Knowledge graph facts not found')

        # TODO: use embeddings.

        raise ValueError('No answer for this question')

    def _process_wh_of_generic_question(self, question: str) -> str:
        question = question.lower().rstrip('?')
        match = re.match(QuestionType.WH_OF.value, question)
        if match is None:
            raise ValueError('No answer for this question')
        predicates = self._knowledge_graph.find_closest_node(match.group(5), predicate=True)
        entities = self._knowledge_graph.find_closest_node(match.group(8))

        logging.debug(f'Search crowd questions')
        if crowd_question := self._knowledge_graph.find_crowd_question(predicates, entities):
            return self._process_crowd_question(crowd_question)
        logging.debug(f'Crowd questions not found')

        query = '''
            PREFIX ddis: <http://ddis.ch/atai/> 
            PREFIX wd: <http://www.wikidata.org/entity/> 
            PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
            PREFIX schema: <http://schema.org/> 
            SELECT DISTINCT ?x WHERE {{ 
                <{e}> <{p}> ?x .
            }}
        '''
        facts = self._knowledge_graph.find_facts(predicates, entities)

        if not facts:
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
            facts = self._knowledge_graph.find_facts(predicates, entities)
        if facts:
            return self._process_facts(facts)
        raise ValueError('No answer for this question')

    def _process_who_of_question(self, question: str) -> str:
        question = question.lower().rstrip('?')
        match = re.match(QuestionType.WHO_OF.value, question)
        if match is None:
            raise ValueError('Invalid question')
        predicates = self._knowledge_graph.find_closest_node(match.group(4), predicate=True)
        entities = self._knowledge_graph.find_closest_node(match.group(7))

        logging.debug(f'Search crowd questions')
        if crowd_question := self._knowledge_graph.find_crowd_question(predicates, entities):
            return self._process_crowd_question(crowd_question)
        logging.debug(f'Crowd questions not found')

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
        facts = self._knowledge_graph.find_facts(predicates, entities)
        if not facts:
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
            facts = self._knowledge_graph.find_facts(predicates, entities)
        if facts:
            return self._process_facts(facts)
        raise ValueError('No answer for this question')

    def _process_crowd_question(self, crowd_question: CrowdQuestion) -> str:
        subject = self._knowledge_graph.get_node_label(crowd_question.subject, is_predicate=False)
        predicate = self._knowledge_graph.get_node_label(crowd_question.predicate, is_predicate=True)
        obj = self._knowledge_graph.get_node_label(crowd_question.object, is_predicate=False)

        n_correct = crowd_question.n_correct
        n_incorrect = crowd_question.n_incorrect
        output = [
            f'{obj} is the {predicate} of {subject}.',
            f'The crowd had an inter-rate agreement of {crowd_question["kappa"]} in this batch.'
        ]
        if n_incorrect > n_correct:
            output.append(f'The crowd thinks this answer is incorrect '
                          '({n_correct} votes for correct and {n_incorrect} votes for incorrect).')
            if crowd_question.mistake_position == 'Predicate':
                output.append(f'Someone has said that the wrong information is {predicate}.')
                if crowd_question.right_label:
                    new_pred = self._knowledge_graph.get_node_label(crowd_question.right_label, is_predicate=True)
                    output.append(f'They think that {obj} is the {new_pred} of {subject}.')
            elif crowd_question.mistake_position == 'Object':
                output.append(f'Someone has said that the wrong information is {obj}.')
                if crowd_question.right_label:
                    new_obj = self._knowledge_graph.get_node_label(crowd_question.right_label, is_predicate=False)
                    output.append(f'They think that {new_obj} is the {predicate} of {subject}.')
            elif crowd_question.mistake_position == 'Subject':
                output.append(f'Someone has said that the wrong information is {subject}.')
                if crowd_question.right_label:
                    new_subj = self._knowledge_graph.get_node_label(crowd_question.right_label, is_predicate=False)
                    output.append(f'They think that {obj} is the {predicate} of {new_subj}.')
        elif n_incorrect <= n_correct:
            output.append(f'The crowd thinks this answer is correct '
                          '({n_correct} votes for correct and {n_incorrect} votes for incorrect).')
        return '\n'.join(output)

    def _process_facts(self, facts: List[Fact]) -> str:
        logging.debug(f'Processing facts: {facts}')
        objs_by_subject_and_pred = defaultdict(list)
        for fact in facts:
            subject, predicate, obj = astuple(fact)
            if self._knowledge_graph.element_is_entity(obj):
                obj = self._knowledge_graph.get_node_label(obj)
            objs_by_subject_and_pred[(subject, predicate)].append(obj)
        output = []
        for (subject, predicate), objects in objs_by_subject_and_pred.items():
            predicate = self._knowledge_graph.get_node_label(predicate, is_predicate=True)
            description = self._knowledge_graph.get_film_description(subject)
            subject = self._knowledge_graph.get_node_label(subject)
            if len(objects) == 1:
                output.append(f'{objects[0]} is the {predicate} of {subject} {description}')
            else:
                output.append(f'The {predicate}s of {subject} {description} are:')
                for obj in objects:
                    output.append(f'\t{obj}')
        return '\n'.join(output)
