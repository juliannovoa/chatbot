import json
import logging
import time
from collections import defaultdict
from typing import Optional

import requests

from src.question_answering_engine import QuestionSolver


class Server:

    @classmethod
    def login(cls, username: str, password: str, url: str) -> requests.Response:
        return requests.post(url=url + "/api/login", json={"username": username, "password": password})

    def __init__(self, username: str, password: str, url: str = 'https://speakeasy.ifi.uzh.ch'):
        self._url = url
        self._user = username
        self._pwd = password
        self._agent_details = None
        self.solver = QuestionSolver()
        self._chatroom_messages = {}

    # check available chat rooms
    def check_rooms(self) -> Optional[requests.Response]:
        try:
            return requests.get(url=self._url + "/api/rooms", params={"session": self._agent_details["sessionToken"]})
        except:
            pass

    # check the state of a chat room
    def check_room_state(self, room_id: str, since: int) -> Optional[requests.Response]:
        try:
            return requests.get(url=self._url + "/api/room/{}/{}".format(room_id, since),
                                params={"roomId": room_id, "since": since,
                                        "session": self._agent_details["sessionToken"]})
        except:
            pass

    # post a message to a chat room
    def post_message(self, room_id: str, message: str) -> None:
        try:
            requests.post(url=self._url + "/api/room/{}".format(room_id),
                          params={"roomId": room_id, "session": self._agent_details["sessionToken"]},
                          data=message)
        except:
            pass

    def run(self) -> None:
        self._agent_details = Server.login(self._user, self._pwd, self._url).json()
        logging.debug("--- agent details:")
        logging.info(json.dumps(self._agent_details, indent=4))

        while True:
            if response := self.check_rooms():
                current_rooms = response.json()["rooms"]
                self._handle_rules(current_rooms)
            time.sleep(3)
            logging.debug("")

    def _handle_rules(self, current_rooms):
        logging.debug("--- {} chatrooms available".format(len(current_rooms)))

        for idx, room in enumerate(current_rooms):
            room_id = room["uid"]
            logging.debug("chat room - {}: {}".format(idx, room_id))

            new_room_state = self.check_room_state(room_id=room_id, since=0)
            if new_room_state is None:
                continue

            new_messages = new_room_state.json()["messages"]
            logging.debug("found {} messages".format(len(new_messages)))

            if room_id not in self._chatroom_messages.keys():
                self._chatroom_messages[room_id] = []

            if len(self._chatroom_messages[room_id]) != len(new_messages):
                for message in new_messages:
                    if message["ordinal"] >= len(self._chatroom_messages[room_id]) and message["session"] != \
                            self._agent_details["sessionId"]:
                        logging.info(f'Question: {message["message"]}')
                        response = self.solver.answer_question(message["message"], room_id)
                        logging.info(f'Response: {response}')
                        self.post_message(room_id=room_id, message=response)

            self._chatroom_messages[room_id] = new_messages
