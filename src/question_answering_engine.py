import logging
import random
import re
from collections import defaultdict
from dataclasses import astuple
from enum import Enum
from typing import Mapping, Callable, List, Optional

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from src import Fact
from src.crowdsourcing import CrowdQuestion
from src.embbedings import Embeddings
from src.knowledge_graph import KnowledgeGraph
from src.multimedia import Multimedia
from src.ner import NameEntityRecognitionModelBERT
from src.utils import remove_stop_words


class QuestionType(Enum):
    GREETING = 0
    MULTIMEDIA = 1
    RECOMMENDATION = 2
    ONE_ENTITY = 3
    TWO_ENTITIES = 4
    CLASS = 5
    UNK = -1


class QuestionSolver:
    EXCUSES = ("Sorry, I didn't get that, could you reformulate your question?",
               "I am sorry. I do not know how to answer this. Could you say it in a different way?",
               "I am not sure if I can help you. Could you ask the question with other words?",
               "I am still learning. Maybe I can help you if you ask in a different manner :)")
    SUGGESTIONS = ("I recommend you these films:",
                   "You should watch these films. I think they are amazing: ",
                   "I think, you will enjoy these films: ",
                   "These are my suggestions: ")
    GREETINGS = (
        "Hallo! I am Skynet. Nice to meet you. I can help you to find information about movies. Please ask me :)",
        "Grüezi! I'm Skynet. Can I help you?",
        "Hello! I'm Skynet. I known a lot about movies. Do you want to ask me something?",
        "Hi! My name is Skynet. I am here to help you :)")
    RECOMMENDATION_KEYWORDS = ("recommend", "recomend", "suggest", "sugest", "similar")
    MULTIMEDIA_KEYWORDS = ("pictur", "imag", "poster", "frame")
    GREETINGS_KEYWORDS = ("hi", "hello", "hallo", "gruezi", "grüezi")
    CLASS_KEYWORDS = ("class", "subclass", "type", "subtyp", "subtype", "instanc", "instance")

    def __init__(self):
        self._ner_model = NameEntityRecognitionModelBERT()
        self._knowledge_graph = KnowledgeGraph()
        self._multimedia = Multimedia(self._knowledge_graph, self._ner_model)
        self._embeddings = Embeddings(self._knowledge_graph)
        self._stop_words = set(stopwords.words('english'))
        self._genres = self._knowledge_graph.find_genres()
        self._questions: Mapping[QuestionType, Callable[[str, str], str]] = {
            QuestionType.MULTIMEDIA: self._process_multimedia,
            QuestionType.RECOMMENDATION: self._process_recommendation,
            QuestionType.GREETING: self._process_greeting,
            QuestionType.ONE_ENTITY: self._process_one_entity_question,
            QuestionType.TWO_ENTITIES: self._process_two_entities_question,
            QuestionType.CLASS: self._process_class_question,
            QuestionType.UNK: self._generate_excuse

        }

    @classmethod
    def _process_greeting(cls, question, idx) -> str:
        return random.choice(cls.GREETINGS)

    @classmethod
    def _generate_excuse(cls, question: Optional[str]= None, id:Optional[str] = None) -> str:
        return random.choice(cls.EXCUSES)

    @classmethod
    def _generate_suggestion_head(cls) -> str:
        return random.choice(cls.SUGGESTIONS)

    def answer_question(self, question: str, idx: Optional[str] = None) -> str:
        question_type = self._get_question_type(question, idx)
        try:
            return self._questions[question_type](question, idx)
        except ValueError:
            return self._generate_excuse()
        except Exception:
            return self._generate_excuse()

    def _get_question_type(self, question: str, idx: Optional[str]) -> QuestionType:
        question = question.rstrip('?')
        # Check if it is a recommendation or multimedia question
        words = word_tokenize(question)
        stemmer = PorterStemmer()
        for word in words:
            stemmed_word = stemmer.stem(word)
            if stemmed_word in self.RECOMMENDATION_KEYWORDS:
                return QuestionType.RECOMMENDATION
            if stemmed_word in self.MULTIMEDIA_KEYWORDS or 'looks like' in question or 'look like' in question:
                return QuestionType.MULTIMEDIA
            if stemmed_word in self.GREETINGS_KEYWORDS:
                return QuestionType.GREETING
            if stemmed_word in self.CLASS_KEYWORDS:
                return QuestionType.CLASS

        # Classify the factual question
        n_entities = len(self._ner_model.find_name_entities(question, idx, update=False))
        if n_entities == 1:
            return QuestionType.ONE_ENTITY
        if n_entities == 2:
            return QuestionType.TWO_ENTITIES

        return QuestionType.UNK

    def _process_class_question(self, question: str, idx) -> str:
        logging.debug(f'Processing class question')
        question = question.rstrip('?')
        ner_result = self._ner_model.find_name_entities(question, idx)

        entities = []
        for entity in ner_result:
            entities.extend(self._knowledge_graph.find_closest_node(entity, predicate=False))

        if not entities:
            question = question.lower()
            question = remove_stop_words(self._stop_words, question)
            for key_word in self.CLASS_KEYWORDS:
                question = re.sub(key_word, '', question)
            entities.extend(self._knowledge_graph.find_closest_node(question))

        logging.debug(f'Search crowd questions')
        if crowd_question := self._knowledge_graph.find_crowd_question(['ddis:indirectSubclassOf'], entities):
            return self._process_crowd_question(crowd_question)
        logging.debug(f'Crowd questions not found')

        logging.debug(f'Search knowledge graph')
        if facts := self._knowledge_graph.get_class(entities):
            return facts
        logging.debug(f'Knowledge graph facts not found')

        logging.debug(f'Search information in embedding')
        if embeddings := self._embeddings.get_closest_solution(['ddis:indirectSubclassOf'], entities):
            return f'Sorry, I have not found the information directly. But I guess that {self._process_facts([embeddings])}'
        logging.debug(f'Knowledge graph facts not found')

        return 'Sorry, I have not found the class.'

    def _process_multimedia(self, question: str, idx) -> str:
        return self._multimedia.process_question(question, idx)

    def _process_recommendation(self, question: str, idx: str) -> str:
        logging.debug(f'Processing recommendation question')
        question = question.rstrip('?')
        ner_result = self._ner_model.find_name_entities(question, idx)

        entities = []
        for entity in ner_result:
            entities.extend(self._knowledge_graph.find_closest_node(entity, predicate=False))

        answer = []
        if suggestion_embeddings := self._embeddings.get_similar_film(entities):
            answer.extend(suggestion_embeddings)
            return self._generate_suggestion_head() + ', '.join(answer) + '.'

        genre = self._find_genre(question)

        if suggestion_graph := self._knowledge_graph.get_recommendations(entities, genre):
            samples = min(5, len(suggestion_graph))
            answer.extend(random.sample(suggestion_graph, samples))
            return self._generate_suggestion_head() + ', '.join(answer) + '.'

        return "I'm sorry, I don't know what to recommend you :("

    def _process_one_entity_question(self, question: str, idx: str) -> str:
        logging.debug(f'Processing one entity question')
        question = question.rstrip('?')
        ner_result = self._ner_model.find_name_entities(question, idx)
        entities = []
        for named_entity in ner_result:
            question = re.sub(named_entity, '', question)
            entities.extend(self._knowledge_graph.find_closest_node(named_entity, predicate=False))
        clean_question = remove_stop_words(self._stop_words, question)

        predicates = self._knowledge_graph.find_closest_node(clean_question, predicate=True)

        logging.debug(f'Search crowd questions')
        if crowd_question := self._knowledge_graph.find_crowd_question(predicates, entities):
            return self._process_crowd_question(crowd_question)
        logging.debug(f'Crowd questions not found')

        logging.debug(f'Search knowledge graph')
        if facts := self._knowledge_graph.find_facts(predicates, entities):
            return self._process_facts(facts)
        logging.debug(f'Knowledge graph facts not found')

        logging.debug(f'Search information in embedding')
        if embeddings := self._embeddings.get_closest_solution(predicates, entities):
            return f'Sorry, I have not found the information directly. But I guess that {self._process_facts([embeddings])}'
        logging.debug(f'Knowledge graph facts not found')

        raise ValueError('No answer for this question')

    def _process_two_entities_question(self, question: str, idx: str) -> str:
        logging.debug(f'Processing two entity question')
        question = question.rstrip('?')
        ner_result = self._ner_model.find_name_entities(question, idx)

        for named_entities in ner_result:
            question = re.sub(named_entities, '', question)
        clean_question = remove_stop_words(self._stop_words, question)
        predicates = self._knowledge_graph.find_closest_node(clean_question, predicate=True)
        entities1 = self._knowledge_graph.find_closest_node(ner_result[0], predicate=False)
        entities2 = self._knowledge_graph.find_closest_node(ner_result[1], predicate=False)

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

        return 'This seems to be wrong. I cannot confirm this fact.'

    def _process_crowd_question(self, crowd_question: CrowdQuestion) -> str:
        subject = self._knowledge_graph.get_node_label(crowd_question.subject, is_predicate=False)
        if crowd_question.predicate == 'ddis:indirectSubclassOf':
            predicate = 'subclass'
        else:
            predicate = self._knowledge_graph.get_node_label(crowd_question.predicate, is_predicate=True)
        obj = self._knowledge_graph.get_node_label(crowd_question.object, is_predicate=False)

        logging.debug(f'Crowd question {crowd_question}')
        n_correct = crowd_question.n_correct
        n_incorrect = crowd_question.n_incorrect
        output = [
            f'{obj} is the {predicate} of {subject}.',
            f'The crowd had an inter-rate agreement of {crowd_question.kappa} in this batch.'
        ]
        if n_incorrect > n_correct:
            output.append(f'The crowd thinks this answer is incorrect '
                          f'({n_correct} votes for correct and {n_incorrect} votes for incorrect).')
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
                          f'({n_correct} votes for correct and {n_incorrect} votes for incorrect).')
        return '\n'.join(output)

    def _process_facts(self, facts: List[Fact]) -> str:
        logging.debug(f'Processing facts: {facts}')
        objs_by_subject_and_pred = defaultdict(list)
        wrong_answers = []
        for fact in facts:
            if checked_answer := self._embeddings.check_triplet(fact):
                wrong_answers.append(checked_answer)
            subject, predicate, obj = astuple(fact)
            if predicate == 'ns2:description':
                return f'{self._knowledge_graph.get_node_label(subject)} is a {obj}.'
            if self._knowledge_graph.element_is_entity(obj):
                obj = self._knowledge_graph.get_node_label(obj)
            objs_by_subject_and_pred[(subject, predicate)].append(obj)
        output = []
        for (subject, predicate), objects in objs_by_subject_and_pred.items():
            if predicate == 'ddis:indirectSubclassOf':
                predicate = 'subclass'
            else:
                predicate = self._knowledge_graph.get_node_label(predicate, is_predicate=True)
            description = self._knowledge_graph.get_film_description(subject)
            subject = self._knowledge_graph.get_node_label(subject)
            if len(objects) == 1:
                output.append(f'{objects[0]} is the {predicate} of {subject}{description}.')
            else:
                new_line = []
                for obj in objects:
                    new_line.append(f'{obj}')
                output.append(f'The {predicate}s of {subject} {description} are: ' + ', '.join(new_line) + '.')
        if wrong_answers:
            output.append('However, I think that my information could be wrong.')
            output.extend(wrong_answers)
        return ' '.join(output)

    def _find_genre(self, question: str) -> Optional[str]:
        for label, genre in self._genres.items():
            if label in question:
                return genre
        return None
