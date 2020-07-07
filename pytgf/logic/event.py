"""
Contains every class related to events and some basic key handlers
"""

import numpy
import json


class Event:
    """
    The top level event type.

    Any event that occurs in the game can be represented using an Event. It can represents the entity interactions with
    the level, or the user external inputs. Custom events can be defined by extending this class. The events are handled
    by the main event queue.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    """

    def __init__(self, tick: int):
        """
        Initializes the Event.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        """

        self.tick = tick


class CancelableEvent(Event):
    """
    A type of event that can be canceled.

    Some events may be canceled during their processing by the event queue. When an CancelableEvent is canceled, it will
    not be processed further by the next handlers of the event queue.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """

    def __init__(self, tick: int):
        """
        Initializes the CancelableEvent.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        """

        Event.__init__(self, tick)

        self._canceled = False

    def cancel(self) -> None:
        """
        Cancels the event.

        By canceling the event, it will not be processed by the other handlers of the EventQueue.
        """

        self._canceled = True

    def is_canceled(self) -> bool:
        """
        Returns whether or not the event got canceled.

        Returns
        -------
        canceled: bool
            The state of the event, True if it got canceled and False otherwise.
        """

        return self._canceled


class EventQueue:
    """
    The event processing core.

    When an event is created, it should be processed by the event queue. The event queue contains the list of different
    user-defined handlers in which the new Event will be passed as argument.

    Attributes
    ---------
    history: list of Event
        The list of fired events.

    Methods
    -------
    register_event_handler(event_type, handler)
        Registers a new event handler.
    fire_event(event)
        Handles a new fired Event.
    """

    DEFAULT_HISTORY_LENGTH = 512

    def __init__(self, history_length: int = DEFAULT_HISTORY_LENGTH):
        """
        Initializes the EventQueue.

        Parameters
        ----------
        history_length: int, optional
            The length of the event history.
        """

        self._history_length = history_length

        self._handlers = []
        self.history = []

    def __len__(self) -> int:
        """
        Returns the length of the event history.

        Returns
        -------
        length: int
            The length of the event history.
        """

        return len(self.history)

    def register_event_handler(self, event_type: type, handler: callable) -> None:
        """
        Registers a new event handler.

        Registers a new handler for the specified event type. Every event that extends from the specified type will be
        passed as argument to the handler when fired. The handler functions should only take one argument, the event
        fired.

        Parameters
        ----------
        event_type: type
            The type of event which will be passed to the handler when fired.
        handler: callable
            The handler function. This function should only take the event passed as argument.
        """

        self._handlers.append((event_type, handler))

    def fire_event(self, event: Event) -> None:
        """
        Handles a new fired event.

        Passes the Event through the different registered handlers for the event type. The orders of registration of the
        handler is taken into account, using a CancelableEvent allow to break out of the handler loop.

        Parameters
        ----------
        event: Event
            The fired event.
        """

        if len(self.history) == self._history_length:
            del self.history[0]

        self.history.append(event)

        for event_type, handler in self._handlers:
            if isinstance(event, event_type):
                if isinstance(event, CancelableEvent):
                    if not event.is_canceled():
                        handler(event)
                else:
                    handler(event)


class Key:
    """
    An enumeration containing the different key codes.

    The user keyboard inputs are represented using key codes. The Key class stores the key codes of every possible input
    that can be issued by the Window.

    Methods
    -------
    get_corresponding_char(key_code)
        Converts the key code into the associated character.
    """

    KEY_BACKSPACE = 65288
    KEY_TAB = 65289
    KEY_LINEFEED = 65290
    KEY_CLEAR = 65291
    KEY_RETURN = 65293
    KEY_ENTER = 65293
    KEY_PAUSE = 65299
    KEY_SCROLL_LOCK = 65300
    KEY_SYS_REQ = 65301
    KEY_ESCAPE = 65307
    KEY_HOME = 65360
    KEY_LEFT = 65361
    KEY_UP = 65362
    KEY_RIGHT = 65363
    KEY_DOWN = 65364
    KEY_PAGE_UP = 65365
    KEY_PAGE_DOWN = 65366
    KEY_END = 65367
    KEY_BEGIN = 65368
    KEY_DELETE = 65535
    KEY_SELECT = 65376
    KEY_PRINT = 65377
    KEY_EXECUTE = 65378
    KEY_INSERT = 65379
    KEY_UNDO = 65381
    KEY_REDO = 65382
    KEY_MENU = 65383
    KEY_FIND = 65384
    KEY_CANCEL = 65385
    KEY_HELP = 65386
    KEY_BREAK = 65387
    KEY_MODE_SWITCH = 65406
    KEY_SCRIPT_SWITCH = 65406
    KEY_MOTION_UP = 65362
    KEY_MOTION_RIGHT = 65363
    KEY_MOTION_DOWN = 65364
    KEY_MOTION_LEFT = 65361
    KEY_MOTION_NEXT_WORD = 1
    KEY_MOTION_PREVIOUS_WORD = 2
    KEY_MOTION_BEGINNING_OF_LINE = 3
    KEY_MOTION_END_OF_LINE = 4
    KEY_MOTION_NEXT_PAGE = 65366
    KEY_MOTION_PREVIOUS_PAGE = 65365
    KEY_MOTION_BEGINNING_OF_FILE = 5
    KEY_MOTION_END_OF_FILE = 6
    KEY_MOTION_BACKSPACE = 65288
    KEY_MOTION_DELETE = 65535
    KEY_NUM_LOCK = 65407
    KEY_NUM_SPACE = 65408
    KEY_NUM_TAB = 65417
    KEY_NUM_ENTER = 65421
    KEY_NUM_F1 = 65425
    KEY_NUM_F2 = 65426
    KEY_NUM_F3 = 65427
    KEY_NUM_F4 = 65428
    KEY_NUM_HOME = 65429
    KEY_NUM_LEFT = 65430
    KEY_NUM_UP = 65431
    KEY_NUM_RIGHT = 65432
    KEY_NUM_DOWN = 65433
    KEY_NUM_PRIOR = 65434
    KEY_NUM_PAGE_UP = 65434
    KEY_NUM_NEXT = 65435
    KEY_NUM_PAGE_DOWN = 65435
    KEY_NUM_END = 65436
    KEY_NUM_BEGIN = 65437
    KEY_NUM_INSERT = 65438
    KEY_NUM_DELETE = 65439
    KEY_NUM_EQUAL = 65469
    KEY_NUM_MULTIPLY = 65450
    KEY_NUM_ADD = 65451
    KEY_NUM_SEPARATOR = 65452
    KEY_NUM_SUBTRACT = 65453
    KEY_NUM_DECIMAL = 65454
    KEY_NUM_DIVIDE = 65455
    KEY_NUM_0 = 65456
    KEY_NUM_1 = 65457
    KEY_NUM_2 = 65458
    KEY_NUM_3 = 65459
    KEY_NUM_4 = 65460
    KEY_NUM_5 = 65461
    KEY_NUM_6 = 65462
    KEY_NUM_7 = 65463
    KEY_NUM_8 = 65464
    KEY_NUM_9 = 65465
    KEY_F1 = 65470
    KEY_F2 = 65471
    KEY_F3 = 65472
    KEY_F4 = 65473
    KEY_F5 = 65474
    KEY_F6 = 65475
    KEY_F7 = 65476
    KEY_F8 = 65477
    KEY_F9 = 65478
    KEY_F10 = 65479
    KEY_F11 = 65480
    KEY_F12 = 65481
    KEY_F13 = 65482
    KEY_F14 = 65483
    KEY_F15 = 65484
    KEY_F16 = 65485
    KEY_LEFT_SHIFT = 65505
    KEY_RIGHT_SHIFT = 65506
    KEY_LEFT_CTRL = 65507
    KEY_RIGHT_CTRL = 65508
    KEY_CAPS_LOCK = 65509
    KEY_LEFT_META = 65511
    KEY_RIGHT_META = 65512
    KEY_LEFT_ALT = 65513
    KEY_RIGHT_ALT = 65514
    KEY_LEFT_WINDOWS = 65515
    KEY_RIGHT_WINDOWS = 65516
    KEY_LEFT_COMMAND = 65517
    KEY_RIGHT_COMMAND = 65518
    KEY_LEFT_OPTION = 65488
    KEY_RIGHT_OPTION = 65489
    KEY_SPACE = 32
    KEY_EXCLAMATION = 33
    KEY_DOUBLE_QUOTE = 34
    KEY_HASH = 35
    KEY_POUND = 35
    KEY_DOLLAR = 36
    KEY_PERCENT = 37
    KEY_AMPERSAND = 38
    KEY_APOSTROPHE = 39
    KEY_LEFT_PARENTHESIS = 40
    KEY_RIGHT_PARENTHESIS = 41
    KEY_ASTERISK = 42
    KEY_PLUS = 43
    KEY_COMMA = 44
    KEY_MINUS = 45
    KEY_PERIOD = 46
    KEY_SLASH = 47
    KEY_COLON = 58
    KEY_SEMICOLON = 59
    KEY_LESS = 60
    KEY_EQUAL = 61
    KEY_GREATER = 62
    KEY_QUESTION = 63
    KEY_AT = 64
    KEY_LEFT_BRACKET = 91
    KEY_BACKSLASH = 92
    KEY_RIGHT_BRACKET = 93
    KEY_ASCII_CIRCUMFLEX = 94
    KEY_UNDERSCORE = 95
    KEY_GRAVE = 96
    KEY_QUOTE_LEFT = 96
    KEY_A = 97
    KEY_B = 98
    KEY_C = 99
    KEY_D = 100
    KEY_E = 101
    KEY_F = 102
    KEY_G = 103
    KEY_H = 104
    KEY_I = 105
    KEY_J = 106
    KEY_K = 107
    KEY_L = 108
    KEY_M = 109
    KEY_N = 110
    KEY_O = 111
    KEY_P = 112
    KEY_Q = 113
    KEY_R = 114
    KEY_S = 115
    KEY_T = 116
    KEY_U = 117
    KEY_V = 118
    KEY_W = 119
    KEY_X = 120
    KEY_Y = 121
    KEY_Z = 122
    KEY_LEFT_BRACE = 123
    KEY_BAR = 124
    KEY_RIGHT_BRACE = 125
    KEY_ASCII_TILDE = 126

    TABLE = {
        KEY_SPACE: " ",
        KEY_NUM_0: "0", KEY_NUM_1: "1", KEY_NUM_2: "2", KEY_NUM_3: "3", KEY_NUM_4: "4",
        KEY_NUM_5: "5", KEY_NUM_6: "6", KEY_NUM_7: "7", KEY_NUM_8: "8", KEY_NUM_9: "9",
        KEY_A: "a", KEY_B: "b", KEY_C: "c", KEY_D: "d", KEY_E: "e", KEY_F: "f", KEY_G: "g", KEY_H: "h", KEY_I: "i",
        KEY_J: "j", KEY_K: "k", KEY_L: "l", KEY_M: "m", KEY_N: "n", KEY_O: "o", KEY_P: "p", KEY_Q: "q", KEY_R: "r",
        KEY_S: "s", KEY_T: "t", KEY_U: "u", KEY_V: "v", KEY_W: "w", KEY_X: "x", KEY_Y: "y", KEY_Z: "z"
    }

    @staticmethod
    def get_corresponding_char(key_code: int) -> str:
        """
        Converts the key code into the associated character.

        Looks up the key code in the class table and finds the corresponding character. If the key code is unknown, it
        returns an empty string.

        Parameters
        ----------
        key_code
            The code of the key.

        Returns
        -------
        character: str
            The character corresponding to the key code.
        """

        if key_code in Key.TABLE:
            return Key.TABLE[key_code]

        return ""


class MouseButton:
    """
    An enumeration containing the different mouse button codes.

    The user mouse button inputs are represented using button codes. The MouseButton class stores the key codes of every
    possible input that can be issued by the Window.
    """

    MOUSE_BUTTON_LEFT = 1
    MOUSE_BUTTON_RIGHT = 4
    MOUSE_BUTTON_MIDDLE = 2


class InputEvent(CancelableEvent):
    """
    A generic type of event used for user input.

    This type of event is used for any event related to a user input. Theses event extend from CancelableEvent, hence
    they can be canceled while being processed by the event queue. This can be useful if more than one object interact
    with the keyboard at once, by giving priority to one over the others.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """
    

class KeyEvent(InputEvent):
    """
    A generic type of event used for key input.

    This type of event is used for any event related to the update of a key input. Theses event extend from
    CancelableEvent, hence they can be canceled while being processed by the event queue. This can be useful if more
    than one object interact with the keyboard at once, by giving priority to one over the others.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    key: int
        The code of the key updated (see also Key).

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """
    
    def __init__(self, tick: int, key: int):
        """
        Initializes the KeyEvent.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        key: int
            The code of the key updated (see also Key).
        """

        CancelableEvent.__init__(self, tick)

        self.key = key


class KeyPressedEvent(KeyEvent):
    """
    The type of KeyEvent used for key press event.

    This kind of event is fired whenever an key is being pressed. Unlike the KeyHeldEvent, this event is only issued
    once before the key got released.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    key: int
        The code of the key pressed.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """


class KeyReleasedEvent(KeyEvent):
    """
    The type of KeyEvent used for key release event.

    This kind of event is fired whenever an key is being released. This event can only be fired after a KeyPressedEvent
    is issued.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    key: int
        The code of the key released.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """


class KeyTypedEvent(KeyEvent):
    """
    The type of KeyEvent used for key type event.

    This kind of event is fired whenever an key is being typed. This event will be fired when the Key is released only
    if it had been held for a short period of time (see InputHandler to specify the delay).

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    key: int
        The code of the key typed.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """


class KeyHeldEvent(KeyEvent):
    """
    The type of KeyEvent used for key hold event.

    This kind of event is fired whenever an key is being held. This event will be repeated each tick while the key is
    being held. It is always preceded by a KeyPressedEvent and followed by KeyReleasedEvent of the same key.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    key: int
        The code of the key held.
    duration: int
        The number of tick for which the key has been held for.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """

    def __init__(self, tick: int, key: int, duration: int):
        """
        Initializes the KeyHeldEvent.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        key: int
            The code of the key held.
        duration: int
            The number of tick for which the key has been held for.
        """

        KeyEvent.__init__(self, tick, key)
        
        self.duration = duration
        

class MouseEvent(InputEvent):
    """
    A generic type of event used for mouse input.

    This type of event is used for any event related to the update of the mouse state (either a mouse button or the
    position of the pointer). Theses events extends from CancelableEvent, hence they can be canceled while being
    processed by the event queue. This can be useful if more than one object interact with the mouse at once, by giving
    priority to one over the others.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    position: numpy.ndarray
        The position of the pointer on the screen.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """
    
    def __init__(self, tick: int, position: numpy.ndarray):
        """
        Initializes the MouseEvent.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        position: numpy.ndarray
            The position of the pointer on the screen.
        """
        
        CancelableEvent.__init__(self, tick)
        
        self.position = position
        

class MouseMovedEvent(MouseEvent):
    """
    The type of MouseEvent used for mouse move event.

    This kind of event is fired whenever the mouse is being moved.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    position: numpy.ndarray
        The position of the pointer on the screen.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """
        

class MouseButtonEvent(MouseEvent):
    """
    A generic type of event used for mouse button input.

    This type of event is used for any event related to the update of a mouse button state. Theses events extends from
    CancelableEvent, hence they can be canceled while being processed by the event queue. This can be useful if more
    than one object interact with the mouse at once, by giving priority to one over the others.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    button: int
        The code of the mouse button updated (see also MouseButton).
    position: numpy.ndarray
        The position of the pointer on the screen.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """

    def __init__(self, tick: int, button: int, position: numpy.ndarray):
        """
        Initializes the MouseButtonEvent.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        button: int
            The code of the mouse button updated.
        position: numpy.ndarray
            The position of the pointer on the screen.
        """
        
        MouseEvent.__init__(self, tick, position)
        
        self.button = button


class MouseButtonPressedEvent(MouseButtonEvent):
    """
    The type of MouseButtonEvent used for mouse button press event.

    This kind of event is fired whenever a mouse button is being pressed. Unlike the MouseDraggedEvent, this event is
    only issued once before the key got released.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    button: int
        The code of the mouse button pressed.
    position: numpy.ndarray
        The position of the pointer on the screen.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """


class MouseButtonReleasedEvent(MouseButtonEvent):
    """
    The type of MouseButtonEvent used for mouse button release event.

    This kind of event is fired whenever a mouse button is being released. This event can only be fired after a
    MouseButtonPressedEvent is issued.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    button: int
        The code of the mouse button released.
    position: numpy.ndarray
        The position of the pointer on the screen.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """


class MouseButtonClickedEvent(MouseButtonEvent):
    """
    The type of MouseButtonEvent used for mouse button click event.

    This kind of event is fired whenever a mouse button is being clicked. This event will be fired when the mouse button
    is released only if it had been held for a short period of time (see InputHandler to specify the delay).

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    button: int
        The code of the mouse button clicked.
    position: numpy.ndarray
        The position of the pointer on the screen.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """


class MouseDraggedEvent(MouseButtonEvent):
    """
    The type of MouseButtonEvent used for mouse drag event.

    This kind of event is fired whenever a mouse button is being dragged (in other words, when the mouse is being moved
    while a mouse button is held). This event will be repeated each tick while the mouse button is being held. It is
    always preceded by a MouseButtonPressedEvent and followed by MouseButtonReleasedEvent of the same mouse button.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    button: int
        The code of the mouse button held.
    position: numpy.ndarray
        The position of the pointer on the screen.
    duration: int
        The number of tick for which the mouse button has been held for.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """

    def __init__(self, tick: int, button: int, position: numpy.ndarray, duration: int):
        """
        Initializes the MouseDraggedEvent.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        button: int
            The code of the mouse button held.
        position: numpy.ndarray
            The position of the pointer on the screen.
        duration: int
            The number of tick for which the mouse button has been held for.
        """

        MouseButtonEvent.__init__(self, tick, button, position)

        self.duration = duration


class InputHandler:
    """
    The source of any user input.

    Any game with user interaction should use an InputHandler.

    Attributes
    ----------
    mouse_position: numpy.ndarray
        The current position of the mouse pointer on the screen.

    Methods
    -------
    fire_events(tick)
        Fires the key and mouse events.
    record()
        Starts the input recording.
    export_record(path)
        Exports the record into a replay file.
    load_replay(path)
        Loads a replay and plays it.
    key_press(key_code)
        Sets the state of the specified key as pressed.
    key_release(key_code)
        Sets the state of the specified key as released.
    mouse_button_press(mouse_button)
        Sets the state of the specified key as pressed.
    mouse_button_release(mouse_button)
        Sets the state of the specified mouse button as released.
    move_mouse(position)
        Sets the position of the mouse pointer.
    """

    DEFAULT_DURATION_TYPE = 20
    DEFAULT_DURATION_CLICK = 20

    def __init__(self, event_queue: EventQueue, duration_type: int = DEFAULT_DURATION_TYPE,
                 duration_click: int = DEFAULT_DURATION_CLICK):
        """
        Initializes the InputHandler.

        Parameters
        ----------
        event_queue: EventQueue
            The main event queue used to handle the events.
        duration_type: int, optional
            The threshold expressed in tick before which a key hold is considered as a type.
        duration_click: int, optional
            the threshold expressed in tick before which a mouse button hold is considered a click.
        """

        self._event_queue = event_queue

        self._duration_type = duration_type
        self._duration_click = duration_click

        self._key_durations = {}
        self._mouse_button_durations = {}

        self.mouse_position = numpy.array((0, 0), dtype=int)
        self._mouse_previous_position = numpy.array((0, 0), dtype=int)

        self._recorder = None
        self._replay = None

        self._should_record = False

    def key_press(self, key_code: int) -> None:
        """
        Sets the state of the specified key as pressed.

        Makes the InputHandler consider the specified key as pressed until another function to release the key is
        called.

        Parameters
        ----------
        key_code: int
            The code of the key pressed.
        """

        if key_code not in self._key_durations:
            self._key_durations[key_code] = [0, False]

    def key_release(self, key_code: int) -> None:
        """
        Sets the state of the specified key as released.

        Makes the InputHandler consider the specified key as released until another function to press the key is called.

        Parameters
        ----------
        key_code: int
            The code of the key released.
        """

        if key_code in self._key_durations:
            self._key_durations[key_code][1] = True

    def mouse_button_press(self, mouse_button: int) -> None:
        """
        Sets the state of the specified mouse button as pressed.

        Makes the InputHandler consider the specified mouse button as pressed until another function to release the
        button is called.

        Parameters
        ----------
        mouse_button: int
            The code of the mouse button pressed.
        """

        if mouse_button not in self._mouse_button_durations:
            self._mouse_button_durations[mouse_button] = [0, False]

    def mouse_button_release(self, mouse_button: int) -> None:
        """
        Sets the state of the specified mouse button as released.

        Makes the InputHandler consider the specified mouse button as released until another function to press the
        button is called.

        Parameters
        ----------
        mouse_button: int
            The code of the mouse button released.
        """

        if mouse_button in self._mouse_button_durations:
            self._mouse_button_durations[mouse_button][1] = True

    def move_mouse(self, position: numpy.ndarray) -> None:
        """
        Sets the position of the mouse pointer.

        Parameters
        ----------
        position: numpy.ndarray
            The current position of the mouse pointer on the screen.
        """

        self.mouse_position = position

    def record(self) -> None:
        """
        Starts the input recording.
        """

        self._should_record = True

    def export_record(self, path) -> None:
        """
        Exports the record into a replay file.

        Parameters
        ----------
        path: str
            The path to the replay file.
        """

        with open(path, "w") as file:
            json.dump(self._recorder.inputs, file)

    def load_replay(self, path, replay_tick: int = 0) -> None:
        """
        Loads a replay and plays it.

        Parameters
        ----------
        path: str
            The path to the replay file.
        replay_tick: int
            The local tick reference of the replay.
        """

        with open(path, "r") as file:
            inputs = json.load(file)

            self._replay = InputReplay(inputs, replay_tick=replay_tick)

    def fire_events(self, tick: int) -> None:
        """
        Fires the key and mouse events.

        This function creates every event related to key and mouse inputs and queues them in the main event queue. It
        should be called at every tick, each time the logic is done.

        Parameters
        ----------
        tick: int
            The current logic tick.
        """

        if self._replay is not None:
            self._replay.play(tick, self)

        if self._should_record:
            self._recorder = InputRecorder(tick)
            self._should_record = False

        for key in list(self._key_durations.keys()):
            if self._key_durations[key][1]:
                self._event_queue.fire_event(KeyReleasedEvent(tick, key))

                if self._recorder is not None:
                    self._recorder.key_release(tick, key)

                if self._key_durations[key][0] <= self._duration_type:
                    self._event_queue.fire_event(KeyTypedEvent(tick, key))

                del self._key_durations[key]
            else:
                if self._key_durations[key][0] == 0:
                    self._event_queue.fire_event(KeyPressedEvent(tick, key))

                    if self._recorder is not None:
                        self._recorder.key_press(tick, key)
                else:
                    self._event_queue.fire_event(KeyHeldEvent(tick, key, self._key_durations[key][0]))

                self._key_durations[key][0] += 1

        for mouse_button in list(self._mouse_button_durations.keys()):
            if self._mouse_button_durations[mouse_button][1]:
                self._event_queue.fire_event(MouseButtonReleasedEvent(tick, mouse_button, self.mouse_position))

                if self._recorder is not None:
                    self._recorder.mouse_button_release(tick, mouse_button)

                if self._mouse_button_durations[mouse_button][0] < self._duration_click:
                    self._event_queue.fire_event(MouseButtonClickedEvent(tick, mouse_button, self.mouse_position))

                del self._mouse_button_durations[mouse_button]
            else:
                if self._mouse_button_durations[mouse_button][0] == 0:
                    self._event_queue.fire_event(MouseButtonPressedEvent(tick, mouse_button, self.mouse_position))

                    if self._recorder is not None:
                        self._recorder.mouse_button_press(tick, mouse_button)
                elif not numpy.array_equal(self.mouse_position, self._mouse_previous_position):
                    self._event_queue.fire_event(MouseDraggedEvent(
                        tick, mouse_button, self.mouse_position, self._mouse_button_durations[mouse_button][0]
                    ))

                self._mouse_button_durations[mouse_button][0] += 1

        if not numpy.array_equal(self.mouse_position, self._mouse_previous_position):
            self._event_queue.fire_event(MouseMovedEvent(tick, self.mouse_position))

            if self._recorder is not None:
                self._recorder.move_mouse(tick, self.mouse_position)

        self._mouse_previous_position = self.mouse_position


class InputReplay:
    """
    A simple way of queueing a sequence of inputs.

    Methods
    -------
    play(tick, input_handler)
        Plays the actions of the replay of the specified tick.
    """

    EVENT_KEY_PRESS = 0
    EVENT_KEY_RELEASE = 1
    EVENT_MOUSE_BUTTON_PRESS = 2
    EVENT_MOUSE_BUTTON_RELEASE = 3
    EVENT_MOUSE_MOVE = 4

    def __init__(self, inputs: dict, replay_tick: int = 0):
        """
        Initializes the InputReplay.

        Parameters
        ----------
        inputs: dict
            The input events with the tick at which they will be issued.
        replay_tick: int, optional
            The local reference tick.
        """

        self._inputs = inputs
        self._tick = replay_tick

    def play(self, tick: int, input_handler: InputHandler) -> None:
        """
        Plays the actions of the replay of the specified tick.

        Parameters
        ----------
        tick: int
            The current logic tick.
        input_handler: InputHandler
            The main input handler controlled by the replay.
        """

        local_tick = str(tick - self._tick)

        if local_tick in self._inputs:
            events = self._inputs[local_tick]

            for event in events:
                if event[0] == InputReplay.EVENT_KEY_PRESS:
                    input_handler.key_press(key_code=event[1])
                elif event[0] == InputReplay.EVENT_KEY_RELEASE:
                    input_handler.key_release(key_code=event[1])
                elif event[0] == InputReplay.EVENT_MOUSE_BUTTON_PRESS:
                    input_handler.mouse_button_press(mouse_button=event[1])
                elif event[0] == InputReplay.EVENT_MOUSE_BUTTON_RELEASE:
                    input_handler.mouse_button_release(mouse_button=event[1])
                elif event[0] == InputReplay.EVENT_MOUSE_MOVE:
                    input_handler.move_mouse(position=numpy.ndarray(event[1]))


class InputRecorder:
    """
    A simple way of recording user inputs.

    When a record is started, it will register the inputs from the user.

    Attributes
    ----------
    inputs: dict
        The input events with the tick at which they were issued.

    Methods
    -------
    key_press(tick, key_code)
        Adds a key press input to the record.
    key_release(tick, key_code)
        Adds a key release input to the record.
    mouse_button_press(tick, mouse_button)
        Adds a mouse button press input to the record.
    mouse_button_release(tick, mouse_button)
        Adds a mouse button release input to the record.
    move_mouse(tick, position)
        Adds a mouse move input to the record.
    get_replay(replay_tick)
        Creates a replay based of the record.
    """

    def __init__(self, record_tick: int = 0):
        """
        Initializes the InputRecorder.

        Parameters
        ----------
        record_tick: int, optional
            The local reference tick.
        """

        self._record_tick = record_tick

        self.inputs = {}

    def _input(self, tick, event_type: int, value: any) -> None:
        """
        Adds a new input to the record.

        Parameters
        ----------
        tick: int
            The tick at which the input was issued.
        event_type: int
            The type of input.
        value: any
            The event data.
        """

        local_tick = tick - self._record_tick

        if local_tick not in self.inputs:
            self.inputs[local_tick] = []

        self.inputs[local_tick].append([event_type, value])

    def key_press(self, tick: int, key_code: int) -> None:
        """
        Adds a key press input to the record.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        key_code: int
            The code of the key pressed.
        """

        self._input(tick, InputReplay.EVENT_KEY_PRESS, key_code)

    def key_release(self, tick: int, key_code: int) -> None:
        """
        Adds a key release input to the record.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        key_code: int
            The code of the key released.
        """

        self._input(tick, InputReplay.EVENT_KEY_RELEASE, key_code)

    def mouse_button_press(self, tick: int, mouse_button: int) -> None:
        """
        Adds a mouse button press input to the record.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        mouse_button: int
            The code of the mouse button pressed.
        """

        self._input(tick, InputReplay.EVENT_MOUSE_BUTTON_PRESS, mouse_button)

    def mouse_button_release(self, tick: int, mouse_button: int) -> None:
        """
        Adds a mouse button release input to the record.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        mouse_button: int
            The code of the mouse button released.
        """

        self._input(tick, InputReplay.EVENT_MOUSE_BUTTON_RELEASE, mouse_button)

    def move_mouse(self, tick: int, position: numpy.ndarray) -> None:
        """
        Adds a mouse move input to the record.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        position: numpy.ndarray
            The current position of the mouse pointer on the screen.
        """

        self._input(tick, InputReplay.EVENT_MOUSE_MOVE, tuple(position))

    def get_replay(self, replay_tick: int = 0) -> InputReplay:
        """
        Creates a replay based of the record.

        Parameters
        ----------
        replay_tick: int
            The local tick reference of the replay.

        Returns
        -------
        replay: InputReplay
            The replay associated to the record.
        """

        return InputReplay(self.inputs, replay_tick)
