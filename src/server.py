import json
import time

import requests


class Server:

    @classmethod
    def login(cls, username: str, password: str, url: str) -> requests.Response:
        return requests.post(url=url + "/api/login", json={"username": username, "password": password})

    def __init__(self, username: str, password: str, url: str):
        self._url = url
        self._agent_details = Server.login(username, password, url).json()

    # check available chat rooms
    def check_rooms(self) -> requests.Response:
        return requests.get(url=self._url + "/api/rooms", params={"session": self._agent_details["sessionToken"]})

    # check the state of a chat room
    def check_room_state(self, room_id: str, since: int) -> requests.Response:
        return requests.get(url=self._url + "/api/room/{}/{}".format(room_id, since),
                            params={"roomId": room_id, "since": since, "session": self._agent_details["sessionToken"]})

    # post a message to a chat room
    def post_message(self, room_id: str, message: str) -> requests.Response:
        return requests.post(url=self._url + "/api/room/{}".format(room_id),
                             params={"roomId": room_id, "session": self._agent_details["sessionToken"]},
                             data=message)

    def run(self) -> None:
        print("--- agent details:")
        print(json.dumps(self._agent_details, indent=4))

        session_token = self._agent_details["sessionToken"]
        chatroom_messages = {}
        while True:
            current_rooms = self.check_rooms().json()["rooms"]
            print("--- {} chatrooms available".format(len(current_rooms)))

            for idx, room in enumerate(current_rooms):
                room_id = room["uid"]
                print("chat room - {}: {}".format(idx, room_id))

                new_room_state = self.check_room_state(room_id=room_id,
                                                       since=0).json()
                new_messages = new_room_state["messages"]
                print("found {} messages".format(len(new_messages)))

                if room_id not in chatroom_messages.keys():
                    chatroom_messages[room_id] = []

                if len(chatroom_messages[room_id]) != len(new_messages):
                    for message in new_messages:
                        if message["ordinal"] >= len(chatroom_messages[room_id]) and message["session"] != \
                                self._agent_details["sessionId"]:
                            response = "Got your message \"{}\" at {}.".format(message["message"],
                                                                               time.strftime("%H:%M:%S, %d-%m-%Y",
                                                                                             time.localtime()))
                            self.post_message(room_id=room_id, message=response)

                chatroom_messages[room_id] = new_messages

            time.sleep(3)
            print("")
