"""
Package containing the classes and functions related to the logic engine. Theses files are the bare minimum for
every project based on the library since they contains the core logic functions.
"""

from pytgf.logic.event import Event, CancelableEvent, EventQueue, Key, MouseButton, InputEvent, KeyEvent, \
    KeyPressedEvent, KeyReleasedEvent, KeyTypedEvent, KeyHeldEvent, MouseEvent, MouseMovedEvent, MouseButtonEvent, \
    MouseButtonPressedEvent, MouseButtonReleasedEvent, MouseButtonClickedEvent, MouseDraggedEvent, InputHandler

from pytgf.logic.physics import AxisAlignedBoundingBox, WorldObject, PhysicsObject, Renderable, Particle, Entity, \
    CollisionMap, TileManager, Direction, CollisionEvent, CollisionWithTileEvent, CollisionWithEntityEvent, QuadTree, \
    WorldUpdater, World, LogicLoop

import numpy


AABB = AxisAlignedBoundingBox


class LogicGame(EventQueue):
    """
    The simplest game object.

    This framework is designed to create a pure logic game (no rendering).

    Attributes
    ----------
    resources: TileManager
        The tile manager used for the world logic.
    world: World
        The game world.
    input_handler: InputHandler
        The main input handler.
    history: list of Event
        The list of fired events.

    Methods
    -------
    update(tick)
        Updates the world and fires the input events.
    run()
        Runs the game logic loop.
    reset()
        Resets the game.
    change_world(tiles, background, logic_area, logic_tile, logic_entity, entity_per_thread, node_capacity, max_depth)
        Creates a new world.
    fire_event(event)
        Handles a new fired Event.
    register_event_handler(event_type, handler)
        Registers a new event handler.
    register_collision_event_handler(handler)
        Registers a new CollisionEvent handler.
    register_collision_tile_event_handler(handler)
        Registers a new CollisionWithTileEvent handler.
    register_collision_entity_event_handler(handler)
        Registers a new CollisionWithEntityEvent handler.
    register_input_event_handler(handler)
        Registers a new InputEvent handler.
    register_key_event_handler(handler)
        Registers a new KeyEvent handler.
    register_key_pressed_event_handler(handler)
        Registers a new KeyPressedEvent handler.
    register_key_released_event_handler(handler)
        Registers a new KeyReleasedEvent handler.
    register_key_typed_event_handler(handler)
        Registers a new KeyTypedEvent handler.
    register_key_held_event_handler(handler)
        Registers a new KeyHeldEvent handler.
    register_mouse_event_handler(handler)
        Registers a new MouseEvent handler.
    register_mouse_moved_event_handler(handler)
        Registers a new MouseMovedEvent handler.
    register_mouse_button_event_handler(handler)
        Registers a new MouseButtonEvent handler.
    register_mouse_button_pressed_event_handler(handler)
        Registers a new MouseButtonPressedEvent handler.
    register_mouse_button_released_event_handler(handler)
        Registers a new MouseButtonReleasedEvent handler.
    register_mouse_button_clicked_event_handler(handler)
        Registers a new MouseButtonClickedEvent handler.
    register_mouse_dragged_event_handler(handler)
        Registers a new MouseDraggedEvent handler.
    """

    def __init__(self, tile_size: int, tick_per_second: float = 60.0, multi_threading: bool = True,
                 safe_mode: bool = True, default_tile_collision_handler: bool = True,
                 default_entity_collision_handler: bool = True):
        """
        Initializes the LogicGame.

        Parameters
        ----------
        tile_size: int
            The size of the tiles expressed in distance units.
        tick_per_second: float, optional
            The tick rate of the loop. It correspond to the number of times the logic will be performed per second.
        multi_threading: bool, optional
            Enables the multi-threading mode if set to True.
        safe_mode: bool, optional
            Enables the update function safe mode (preventing infinite loop) if set to True.
        default_tile_collision_handler: bool, optional
            Registers the default tile collision event handler if set to True.
        default_entity_collision_handler: bool, optional
            Registers the default entity collision event handler if set to True.
        """

        EventQueue.__init__(self)

        if not hasattr(self, "input_handler"):
            self.input_handler = InputHandler(self)

        if not hasattr(self, "resources"):
            self.resources = TileManager(tile_size)

        if not hasattr(self, "_loop"):
            self._loop = LogicLoop(tick_per_second, self.update)

        self._multi_threading = multi_threading
        self._safe_mode = safe_mode

        if default_tile_collision_handler:
            self.register_collision_tile_event_handler(CollisionWithTileEvent.default_handler_collision_tile)

        if default_entity_collision_handler:
            self.register_collision_entity_event_handler(CollisionWithEntityEvent.default_handler_collision_entity)

        self.world = None

    def update(self, tick: int) -> None:
        """
        Updates the world and fires the input events.

        Parameters
        ----------
        tick: int
            The current logic tick.
        """

        if self.world is not None:
            self.world.update(tick)

        self.input_handler.fire_events(tick)

    def run(self, number_of_ticks: int = None, max_speed: bool = False) -> None:
        """
        Runs the game logic loop.

        Parameters
        ----------
        number_of_ticks: int, optional
            The number of ticks over which the logic will be performed. Il this parameter is not set, the loop will run
            until the stop function is called.
        max_speed: bool, optional
            Ignores the tick rate and runs at maximum speed if set to True.
        """

        self._loop(number_of_ticks=number_of_ticks, max_speed=max_speed)

    def stop(self) -> None:
        """
        Stops the game logic loop.
        """

        self._loop.stop()

    def reset(self) -> None:
        """
        Resets the game (this function is called on the initialization).
        """

    def change_world(self, tiles: numpy.ndarray, background: str, logic_area: AxisAlignedBoundingBox = None,
                     logic_tile: bool = True, logic_entity: bool = True,
                     entity_per_thread: int = WorldUpdater.DEFAULT_ENTITY_PER_THREAD,
                     node_capacity: int = QuadTree.DEFAULT_NODE_CAPACITY,
                     max_depth: int = QuadTree.DEFAULT_MAX_DEPTH) -> None:
        """
        Creates a new world.

        Creates a world and sets it as the played one, by using the local logic objects of the game.

        Parameters
        ----------
        tiles: numpy.ndarray
            The tiles array of the level.
        background: str
            The background name.
        logic_area: AxisAlignedBoundingBox, optional
            The area over which the logic is performed.
        logic_tile: bool, optional
            Enables the collision detection with the tiles if set to True.
        logic_entity: bool, optional
            Enables the collision detection with the entities is set to True.
        entity_per_thread: int, optional
            The number of entity in each thread.
        node_capacity: int, optional
            The object capacity of the leaf before it divides into smaller leaves.
        max_depth: int, optional
            The maximum depth of the tree.
        """

        self.world = World(
            self.resources, self, tiles, background, logic_area=logic_area, logic_tile=logic_tile,
            logic_entity=logic_entity, multi_threading=self._multi_threading, safe_mode=self._safe_mode,
            entity_per_thread=entity_per_thread, node_capacity=node_capacity, max_depth=max_depth
        )

    def register_collision_event_handler(self, handler: callable) -> None:
        """
        Registers a new CollisionEvent handler.

        Registers a new handler for any event related to a collision. Every other event related to collisions such as
        CollisionWithTileEvent and CollisionWithEntityEvent will be processed by the handler.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the CollisionEvent passed as argument.
        """

        self.register_event_handler(CollisionEvent, handler)

    def register_collision_tile_event_handler(self, handler: callable) -> None:
        """
        Registers a new CollisionWithTileEvent handler.

        Registers a new handler for any event related to a collision with a tile.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the CollisionWithTileEvent passed as argument.
        """

        self.register_event_handler(CollisionWithTileEvent, handler)

    def register_collision_entity_event_handler(self, handler: callable) -> None:
        """
        Registers a new CollisionWithEntityEvent handler.

        Registers a new handler for any event related to a collision with a entity.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the CollisionWithEntityEvent passed as argument.
        """

        self.register_event_handler(CollisionWithEntityEvent, handler)

    def register_input_event_handler(self, handler: callable) -> None:
        """
        Registers a new InputEvent handler.

        Registers a new handler for any event related to a user input. Every other event related to inputs like key and
        mouse related events will be processed by the handler.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the InputEvent passed as argument.
        """

        self.register_event_handler(InputEvent, handler)

    def register_key_event_handler(self, handler: callable) -> None:
        """
        Registers a new KeyEvent handler.

        Registers a new handler for any event related to a key update. Every other event related to keys, such as
        KeyPressedEvent, KeyReleasedEvent, KeyTypedEvent and KeyHeldEvent will be processed by the handler.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the KeyEvent passed as argument.
        """

        self.register_event_handler(KeyEvent, handler)

    def register_key_pressed_event_handler(self, handler: callable) -> None:
        """
        Registers a new KeyPressedEvent handler.

        Registers a new handler for any event related to a key press.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the KeyPressedEvent passed as argument.
        """

        self.register_event_handler(KeyPressedEvent, handler)

    def register_key_released_event_handler(self, handler: callable) -> None:
        """
        Registers a new KeyReleasedEvent handler.

        Registers a new handler for any event related to a key release.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the KeyReleasedEvent passed as argument.
        """

        self.register_event_handler(KeyReleasedEvent, handler)

    def register_key_typed_event_handler(self, handler: callable) -> None:
        """
        Registers a new KeyTypedEvent handler.

        Registers a new handler for any event related to a key type.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the KeyTypedEvent passed as argument.
        """

        self.register_event_handler(KeyTypedEvent, handler)

    def register_key_held_event_handler(self, handler: callable) -> None:
        """
        Registers a new KeyHeldEvent handler.

        Registers a new handler for any event related to a key hold.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the KeyHeldEvent passed as argument.
        """

        self.register_event_handler(KeyHeldEvent, handler)

    def register_mouse_event_handler(self, handler: callable) -> None:
        """
        Registers a new MouseEvent handler.

        Registers a new handler for any event related to a mouse update (either a mouse button or the position of the
        pointer). Every other event related to mouse, such as MouseMovedEvent, MouseButtonPressedEvent,
        MouseButtonReleasedEvent, MouseButtonClickedEvent and MouseDraggedEvent will be processed by the handler.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the MouseEvent passed as argument.
        """

        self.register_event_handler(MouseEvent, handler)

    def register_mouse_moved_event_handler(self, handler: callable) -> None:
        """
        Registers a new MouseMovedEvent handler.

        Registers a new handler for any event related to mouse move.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the MouseMovedEvent passed as argument.
        """

        self.register_event_handler(MouseMovedEvent, handler)

    def register_mouse_button_event_handler(self, handler: callable) -> None:
        """
        Registers a new MouseButtonEvent handler.

        Registers a new handler for any event related to a mouse button update. Every other event related to mouse, such
        as MouseButtonPressedEvent, MouseButtonReleasedEvent, MouseButtonClickedEvent and MouseDraggedEvent will be
        processed by the handler.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the MouseButtonEvent passed as argument.
        """

        self.register_event_handler(MouseButtonEvent, handler)

    def register_mouse_button_pressed_event_handler(self, handler: callable) -> None:
        """
        Registers a new MouseButtonPressedEvent handler.

        Registers a new handler for any event related to a mouse button press.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the MouseButtonPressedEvent passed as argument.
        """

        self.register_event_handler(MouseButtonPressedEvent, handler)

    def register_mouse_button_released_event_handler(self, handler: callable) -> None:
        """
        Registers a new MouseButtonReleasedEvent handler.

        Registers a new handler for any event related to a mouse button release.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the MouseButtonReleasedEvent passed as argument.
        """

        self.register_event_handler(MouseButtonReleasedEvent, handler)

    def register_mouse_button_clicked_event_handler(self, handler: callable) -> None:
        """
        Registers a new MouseButtonClickedEvent handler.

        Registers a new handler for any event related to a mouse button click.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the MouseButtonClickedEvent passed as argument.
        """

        self.register_event_handler(MouseButtonClickedEvent, handler)

    def register_mouse_dragged_event_handler(self, handler: callable) -> None:
        """
        Registers a new MouseDraggedEvent handler.

        Registers a new handler for any event related to a mouse drag.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the MouseDraggedEvent passed as argument.
        """

        self.register_event_handler(MouseDraggedEvent, handler)

    def __del__(self) -> None:
        """
        Closes the game (this function should clean up the memory).
        """

        self.stop()

    def __str__(self) -> str:
        """
        Returns a description string of the object.

        Returns
        -------
        string: str
            The string object description.
        """

        return "LogicGame[input_handler=" + str(self.input_handler) + ", resources=" + str(self.resources) + ", " + \
               "world=" + str(self.world) + ", multi_threading=" + str(self._multi_threading) + ", " + \
               "safe_mode=" + str(self._safe_mode) + "]"
