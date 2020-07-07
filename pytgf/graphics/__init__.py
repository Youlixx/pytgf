"""
Package containing the classes and functions related to the graphics engine. Theses files are not necessary to run a
game without rendering.
"""

from pytgf.graphics.graphics import Camera, Texture, ShaderProgram, SpriteSet, TileSet, ResourceManager, RenderLoop, \
    WorldRenderer

from pytgf.graphics.gui import GUIFont, GUIBorder, GUIComponent, GUILayout, GUIAbsoluteLayout, GUIListLayout, \
    GUIContainer, GUILabel, GUIImage, GUITextField, GUIEvent, GUIFocusedEvent, GUIUnfocusedEvent, GUIManager

from pytgf.logic import AxisAlignedBoundingBox, LogicGame, QuadTree, WorldUpdater, World

import numpy


class Game(LogicGame):
    """
    A standard game object.

    This framework is designed to create a complete game (logic and rendering). This is supposed to be used in
    windowless mode (standalone), see WindowedGame for windowed mode.

    Attributes
    ----------
    resources: ResourceManager
        The main resource manager used for the rendering.
    world: World
        The game world.
    input_handler: InputHandler
        The main input handler.
    camera: Camera
        The camera used for the rendering.
    gui: GUIManager
        The main GUI container of the game.
    history: list of Event
        The list of fired events.

    Methods
    -------
    update(tick)
        Updates the world and fires the input events.
    render(frame)
        Renders the world and the GUI.
    run()
        Runs the game logic loop.
    reset()
        Resets the game.
    close()
        Closes the game.
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
    register_gui_event_handler(handler)
        Registers a new GUIEvent handler.
    register_gui_focused_event_handler(handler)
        Registers a new GUIFocusedEvent handler.
    register_gui_unfocused_event_handler(handler)
        Registers a new GUIUnfocusedEvent handler.
    """

    def __init__(self, tile_size: int, viewport: [tuple, numpy.ndarray], scale: float = 1.0, standalone: bool = True,
                 width: int = 800, height: int = 600, glsl_version: int = 330, shader_world: tuple = None,
                 shader_sprite: tuple = None, tick_per_second: float = 60, frame_per_second: float = 60,
                 multi_threading: bool = True, default_tile_collision_handler: bool = True,
                 default_entity_collision_handler: bool = True, default_gui_handler: bool = True):
        """
        Initializes the Game.

        Parameters
        ----------
        tile_size: int
            The size of the tiles expressed in distance units.
        viewport: [tuple, numpy.ndarray]
            The size of the field of view.
        scale: float, optional
            The rendering scale factor.
        standalone: bool, optional
            This have to be set to True if there is no window used. It will create a virtual window to render the game.
        width: int, optional
            Used in standalone mode, the width of the virtual window.
        height: int, optional
            Used is standalone mode, the height of the virtual window.
        glsl_version: int, optional
            The required GLSL version.
        shader_world: tuple of strings, optional
            The couple of source code for the fragment and vertex shaders used for the rendering of the world.
        shader_sprite: tuple of strings, optional
            The couple of source code for the fragment and vertex shaders used for the rendering of sprites.
        event_queue: EventQueue, optional
            The main event queue used to handle the events.
        tick_per_second: float, optional
            The tick rate of the loop. It correspond to the number of times the logic will be performed per second.
        frame_per_second: float, optional
            The frame rate of the loop. It correspond to the number of times the frame will be rendered per second.
        multi_threading: bool, optional
            Enables the multi-threading mode if set to True.
        default_tile_collision_handler: bool, optional
            Registers the default tile collision event handler if set to True.
        default_entity_collision_handler: bool, optional
            Registers the default entity collision event handler if set to True.
        default_gui_handler: bool: optional
            Registers the default GUI event handler if set to True.
        """

        if not hasattr(self, "resources"):
            self.resources = ResourceManager(
                tile_size, scale=scale, standalone=standalone, width=width, height=height, glsl_version=glsl_version,
                shader_world=shader_world, shader_sprite=shader_sprite
            )

        if not hasattr(self, "_loop"):
            self._loop = RenderLoop(tick_per_second, frame_per_second, self.update, self.render, self.close)

        self.camera = Camera(viewport)

        self.gui = GUIManager(self.resources, self, self.camera.viewport)

        self._world_renderer = None

        LogicGame.__init__(
            self, tile_size, tick_per_second=tick_per_second, multi_threading=multi_threading,
            default_tile_collision_handler=default_tile_collision_handler,
            default_entity_collision_handler=default_entity_collision_handler
        )

        if default_gui_handler:
            self.register_mouse_button_clicked_event_handler(self.gui.default_handler_gui_click)

    def update(self, tick: int) -> None:
        """
        Updates the world and fires the input events.

        Parameters
        ----------
        tick: int
            The current logic tick.
        """

        super().update(tick)

        self.gui.fire_events(tick)

    def render(self, frame: int) -> None:
        """
        Renders the world and the GUI.

        Parameters
        ----------
        frame: int
            The current frame.
        """

        if self._world_renderer is not None:
            self._world_renderer.render(self.camera)

        self.gui.render()

    def change_world(self, tiles: numpy.ndarray, background: str, logic_area: AxisAlignedBoundingBox = None,
                     logic_tile: bool = True, logic_entity: bool = True, safe_mode: bool = True,
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
        safe_mode: bool, optional
            Enables the update function safe mode (preventing infinite loop) if set to True.
        entity_per_thread: int, optional
            The number of entity in each thread.
        node_capacity: int, optional
            The object capacity of the leaf before it divides into smaller leaves.
        max_depth: int, optional
            The maximum depth of the tree.
        """

        super().change_world(
            tiles, background, logic_area=logic_area, logic_tile=logic_tile, logic_entity=logic_entity,
            safe_mode=safe_mode, entity_per_thread=entity_per_thread, node_capacity=node_capacity, max_depth=max_depth
        )

        self._world_renderer = WorldRenderer(self.resources, self.world)

    def register_gui_event_handler(self, handler: callable) -> None:
        """
        Registers a new GUIEvent handler.

        Registers a new handler for any event related to a GUI update. Every other event related to GUI, such as
        GUIFocusedEvent and GUIUnfocusedEvent will be processed by the handler.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the GUIEvent passed as argument.
        """

        self.register_event_handler(GUIEvent, handler)

    def register_gui_focused_event_handler(self, handler: callable) -> None:
        """
        Registers a new GUIFocusedEvent handler.

        Registers a new handler for any event related to a component getting focused.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the GUIFocusedEvent passed as argument.
        """

        self.register_event_handler(GUIFocusedEvent, handler)

    def register_gui_unfocused_event_handler(self, handler: callable) -> None:
        """
        Registers a new GUIUnfocusedEvent handler.

        Registers a new handler for any event related to a component getting unfocused.

        Parameters
        ----------
        handler: callable
            The handler function. This function should only take the GUIUnfocusedEvent passed as argument.
        """

        self.register_event_handler(GUIUnfocusedEvent, handler)
