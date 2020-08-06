"""
Contains every class related to windows
"""

from pytgf.logic.event import EventQueue, InputHandler
from pytgf.graphics.graphics import RenderLoop, ResourceManager

import pyglet
import numpy


class Wrapper(pyglet.window.Window):
    """
    A custom pyglet window object

    This class overrides some default methods that would trigger OpenGL errors.
    """

    def on_resize(self, width: int, height: int) -> None:
        """
        Overrides the resize callback function which triggers errors.

        Parameters
        ----------
        width: int
            The new width of the window.
        height: int
            The new height of the window.
        """

    def on_draw(self) -> None:
        """
        Overrides the default drawing routine to avoid OpenGL errors.
        """


class Window(InputHandler):
    """
    A generic window object.

    A extended version of the default input handler to fire the window events.

    Attributes
    ----------
    event_queue: EventQueue
        The main event queue used to handle the events.
    mouse_position: numpy.ndarray
        The current position of the mouse pointer on the screen.
    should_close: bool
        Closes the window if set to True.

    Methods
    -------
    render()
        Swaps the buffers and renders the latest drawn frame by OpenGL.
    fire_events(tick)
        Fires the key and mouse events.
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

    def __init__(self, title: str, event_queue: EventQueue, width: int = 800, height: int = 600,
                 full_screen: bool = True, hide_cursor: bool = False):
        """
        Initializes the Window.

        Parameters
        ----------
        title: str
            The name of the window.
        event_queue: EventQueue
            The main event queue used to handle the events.
        width: int, optional
            The width of the window if not created in full screen.
        height: int, optional
            The height of the window if not created in full screen.
        full_screen: bool, optional
            Creates the a full screen window if set to True.
        hide_cursor: bool, optional
            Hides the cursor in the window area if set to True.
        """

        InputHandler.__init__(self, event_queue)

        config = pyglet.gl.Config(
            major_version=3,
            minor_version=3,
            forward_compatible=True,
            depth_size=24,
            double_buffer=True,
            sample_buffers=0,
            samples=0
        )

        if full_screen:
            display = pyglet.canvas.get_display()
            screen = display.get_default_screen()
            self._width, self._height = screen.width, screen.height
        else:
            self._width = width
            self._height = height

        self._window = Wrapper(
            width=self._width, height=self._height,
            caption=title,
            resizable=False,
            vsync=False,
            fullscreen=full_screen,
            config=config
        )

        self._window.set_mouse_visible(not hide_cursor)

        self._window.on_key_press = self._on_key_pressed
        self._window.on_key_release = self._on_key_released
        self._window.on_mouse_press = self._on_mouse_pressed
        self._window.on_mouse_release = self._on_mouse_released
        self._window.on_mouse_motion = self._on_mouse_moved
        self._window.on_mouse_drag = self._on_mouse_dragged
        self._window.on_close = self._on_close

        self.should_close = False

    def _on_key_pressed(self, symbol: int, modifiers: int) -> None:
        """
        Callback function adding the key press events to the queue

        Parameters
        ----------
        symbol: int
            The code of the pressed key.
        modifiers: int
            A bitwise combination of the active key modifiers (unused).
        """

        self.key_press(symbol)

    def _on_key_released(self, symbol: int, modifiers: int) -> None:
        """
        Callback function adding the key release events to the queue

        Parameters
        ----------
        symbol: int
            The code of the released key.
        modifiers: int
            A bitwise combination of the active key modifiers (unused).
        """

        self.key_release(symbol)

    def _on_mouse_pressed(self, x: int, y: int, button: int, modifiers: int) -> None:
        """
        Callback function adding the mouse button press events to the queue

        Parameters
        ----------
        x: int
            The position of the mouse pointer along the x axis (unused).
        y: int
            The position of the mouse pointer along the y axis (unused).
        button: int
            The code of the pressed mouse button.
        modifiers:
            A bitwise combination of the active key modifiers (unused).
        """

        self.mouse_button_press(button)

    def _on_mouse_released(self, x: int, y: int, button: int, modifiers: int) -> None:
        """
        Callback function adding the mouse button release events to the queue

        Parameters
        ----------
        x: int
            The position of the mouse pointer along the x axis (unused).
        y: int
            The position of the mouse pointer along the y axis (unused).
        button: int
            The code of the released mouse button.
        modifiers:
            A bitwise combination of the active key modifiers (unused).
        """

        self.mouse_button_release(button)

    def _on_mouse_moved(self, x: int, y: int, dx: int, dy: int) -> None:
        """
        Callback function updating the position of the mouse

        Parameters
        ----------
        x: int
            The position of the mouse pointer along the x axis.
        y: int
            The position of the mouse pointer along the y axis.
        dx: int
            The displacement of the mouse pointer along the x axis (unused).
        dy: int
            The displacement of the mouse pointer along the y axis (unused).
        """

        if 0 <= x <= self._width and 0 <= y <= self._height:
            x = 2 * (x - self._width / 2) / self._width
            y = 2 * (y - self._height / 2) / self._height

            self.move_mouse(numpy.array((x, y), dtype=float))

    def _on_mouse_dragged(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        """
        Callback function updating the position of the mouse

        Parameters
        ----------
        x: int
            The position of the mouse pointer along the x axis.
        y: int
            The position of the mouse pointer along the y axis.
        dx: int
            The displacement of the mouse pointer along the x axis (unused).
        dy: int
            The displacement of the mouse pointer along the y axis (unused).
        button: int
            The code of the released mouse button. (unused)
        modifiers:
            A bitwise combination of the active key modifiers (unused).
        """

        if 0 <= x <= self._width and 0 <= y <= self._height:
            x = 2 * (x - self._width / 2) / self._width
            y = 2 * (y - self._height / 2) / self._height

            self.move_mouse(numpy.array((x, y), dtype=float))

    def _on_close(self) -> None:
        """
        Callback function called when the window is closing.
        """

        self.should_close = True
        self._window.close()

    def render(self) -> None:
        """
        Swaps the buffers and renders the latest drawn frame by OpenGL.
        """

        if not self.should_close:
            self._window.flip()

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

        self._window.dispatch_events()

        super().fire_events(tick)

    def __str__(self) -> str:
        """
        Returns a description string of the object.

        Returns
        -------
        string: str
            The string object description.
        """

        return "Window[width=" + str(self._width) + ", height=" + str(self._height) + ", " + \
               "should_close=" + str(self.should_close) + "]"


class AssetManager(ResourceManager):
    """
    The asset resource manager.

    An asset manager is a complete resource manager. It includes the physics of the tiles, the textures used for the
    rendering and the audio sources to play music and sound effects. If there is no window used for the rendering, this
    should be created in standalone mode. It contains the main OpenGL context as well as the different graphical
    graphics used for the rendering.

    Attributes
    ----------
    context: moderngl.Context
        The main OpenGL context.
    tile_size: int
        The size of the tiles expressed in distance units.
    scale: float
        The rendering scale factor.
    shader_world: ShaderProgram
        The shader program used for the rendering of the world.
    shader_sprite: ShaderProgram
        The shader program used for the rendering of sprites.
    model_world: Model
        The model used to render the world.
    model_sprite: Model
        The model used to render sprites.
    tile_set: TileSet
        The tile set used for the world rendering.
    sprite_sets: dict of SpriteSet
        The sprite sets used for the sprite rendering.
    volume: float
        The global audio gain volume.

    Methods
    -------
    update()
        Updates the internal clock.
    register_music(name, path)
        Registers a new music.
    register_effect(name, path)
        Registers a new sound effect.
    register_collision_map(collision_map)
        Registers a new collision map.
    register_tile(id_collision_map, id_texture)
        Registers a new tile.
    register_tile_set(texture, width, height)
        Registers the main tile set.
    register_background(name)
        Registers a new background.
    register_background_texture(texture)
        Registers a new background texture.
    register_background_layer(name, id_texture, ratio_x, ratio_y)
        Registers a new background layer.
    register_sprite_set(sprite_set)
        Registers a new sprite set.
    register_sprite_texture(sprite_set, texture)
        Registers a new sprite texture.
    register_sprite_animation(sprite_set, period, animation)
        Registers a new sprite animation.
    get_collision_map(id_tile)
        Returns the collision map associated to the tile.
    get_id_texture(id_tile)
        Returns the texture associated to the tile.
    get_last_frame_buffer()
        Returns the latest rendered frame.
    render_background(name, camera)
        Renders the background.
    render_sprite(renderable, camera)
        Renders a sprite.
    increment_animation_pointer(renderable)
        Increments the animation pointer of the renderable.
    play_music(name)
        Plays a music.
    play_effect(name, volume)
        Plays a sound effect.
    clear()
        Clears the music player.
    """

    def __init__(self, tile_size: int, scale: float = 1.0, standalone: bool = False, width: int = 800,
                 height: int = 600, glsl_version: int = 330, shader_world: tuple = None, shader_sprite: tuple = None,
                 sampling_rate: float = 60):
        """
        Initializes the AudioManager.

        Parameters
        ----------
        tile_size: int
            The size of the tiles expressed in distance units.
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
        sampling_rate: float, optional
            The rate at which the audio is updated.
        """

        ResourceManager.__init__(
            self, tile_size, scale=scale, standalone=standalone, width=width, height=height, glsl_version=glsl_version,
            shader_world=shader_world, shader_sprite=shader_sprite
        )

        pyglet.options['audio'] = ('openal', 'pulse', 'directsound', 'silent')

        self._sampling_rate = sampling_rate

        self._musics = {}
        self._effects = {}

        self._volume = 1.0

        self._music_player = pyglet.media.Player()
        self._music_volume = 1.0
        self._music_stack = 0
        self._music_played = None

    @property
    def volume(self) -> float:
        """
        The volume property representing the global volume of the sounds played.
        """

        return self._volume

    @volume.setter
    def volume(self, volume: float) -> None:
        """
        Setter function for the audio volume.

        Parameters
        ----------
        volume: float
            The global volume of the sounds played.
        """

        self._volume = volume

        if self._music_player is not None:
            self._music_player.volume = volume * self._music_volume

    def update(self) -> None:
        """
        Updates the internal clock.
        """

        if self._music_player is None:
            return

        pyglet.clock.tick()

        if self._music_stack > 0:
            self._music_stack -= 1

            if self._music_stack <= 0:
                self.play_music(self._music_played, self._music_volume)

    def register_music(self, name: str, path: str) -> None:
        """
        Registers a new music.

        Only a single music can be played at once. The audio file is streamed so large files don't have to be stored in
        the memory.

        Parameters
        ----------
        name: str
            The name of the music.
        path: str
            The path to the audio file (RIFF/WAV file, unless FFmpeg is installed).
        """

        self._musics[name] = path

    def register_effect(self, name: str, path: str) -> None:
        """
        Registers a new sound effect.

        Multiple sound effects can be played at once. The audio file is stored in the memory to ease the CPU usage,
        hence the sound effects should be as light as possible.

        Parameters
        ----------
        name: str
            The name of the sound effect.
        path: str
            The path to the audio file (RIFF/WAV file, unless FFmpeg is installed).
        """

        self._effects[name] = pyglet.media.load(path, streaming=False)

    def play_music(self, name: str, volume: float = 1.0) -> None:
        """
        Plays a music.

        Parameters
        ----------
        name: str
            The name of the music to play.
        volume: float, optional
            The volume of the music.
        """

        source = pyglet.media.load(self._musics[name])

        self._music_player = pyglet.media.Player()
        self._music_player.queue(source)
        self._music_player.volume = self._volume * volume
        self._music_player.play()

        self._music_volume = volume
        self._music_played = name
        self._music_stack = source.duration * self._sampling_rate

    def play_effect(self, name, volume: float = 1.0) -> None:
        """
        Plays a sound effect.

        Parameters
        ----------
        name: str
            The name of the sound effect to play.
        volume: float, optional
            The volume of the sound effect.
        """

        player = self._effects[name].play()
        player.volume = self._volume * volume


class WindowRenderLoop(RenderLoop):
    """
    The main game loop function with addition of rendering routine for windowed mode.

    This object is callable, hence is should be called as if it was a function. When called, the loop starts and run the
    logic function at each tick, at a specified tick rate and run the render function at each frame, at a specified
    frame rate.

    Methods
    -------
    stop()
        Stops the main loop.
    """

    def __init__(self, tick_per_second: float, frame_per_second: float, function_logic: callable,
                 function_render: callable, window: Window):
        """
        Initializes the WindowRenderLoop

        Parameters
        ----------
        tick_per_second: float
            The tick rate of the loop. It correspond to the number of times the logic will be performed per second.
        frame_per_second: float
            The frame rate of the loop. It correspond to the number of times the frame will be rendered per second.
        function_logic: callable
            The logic function of the loop called each tick.
        function_render: callable
            The render function of the loop called each frame.
        window: Window
            The window in which the game is displayed.
        """

        RenderLoop.__init__(self, tick_per_second, frame_per_second, function_logic, function_render)

        self._window = window

    def _do_render(self) -> None:
        """
        Calls the render routine.
        """

        super()._do_render()

        self._window.render()

    def __str__(self) -> str:
        """
        Returns a description string of the object.

        Returns
        -------
        string: str
            The string object description.
        """

        return "RenderLoop[tick=" + str(self._tick) + ", frame=" + str(self._frame) + ", " + \
               "window=" + str(self._window) + ", running=" + str(self._running) + "]"
