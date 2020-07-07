"""
Contains every classes related to the graphics engine.
"""

from pytgf.logic.physics import Renderable, TileManager, World, LogicLoop, array_format

from time import sleep

import numpy
import moderngl

DEFAULT_SHADER_SPRITE_VERTEX = """#version 330

uniform mat4 position;
uniform mat4 projection;

in vec2 vertices;
in vec2 textures;

out vec2 textureCoordinates;

void main() {
    textureCoordinates = textures;

    gl_Position = projection * position * vec4(vertices, 0.0, 1.0);
}
"""

DEFAULT_SHADER_SPRITE_FRAGMENT = """#version 330

uniform sampler2D sampler;

in vec2 textureCoordinates;
out vec4 outColor;

void main() {
    outColor = texture(sampler, textureCoordinates);
}
"""

DEFAULT_SHADER_WORLD_VERTEX = """#version 330

uniform mat4 projection;
uniform mat4 position;

in vec2 vertices;
in vec2 textures;

out vec2 textureCoordinates;

void main() {
    textureCoordinates = textures;

    gl_Position = projection * position * vec4(vertices, 0.0, 1.0);
}
"""

DEFAULT_SHADER_WORLD_FRAGMENT = """#version 330

uniform sampler2D levelTexture;
uniform sampler2D tilesTexture;

uniform int levelWidth;
uniform int levelHeight;

uniform int tilesWidth;
uniform int tilesHeight;

in vec2 textureCoordinates;
out vec4 outColor;

void main() {
    vec2 coordinates = vec2(textureCoordinates.xy);
    
    if(textureCoordinates.y * levelHeight == floor(coordinates.y * levelHeight)) {
        coordinates = vec2(coordinates.x, coordinates.y + 1.0 / (levelHeight * levelHeight));
    }
    
    if(textureCoordinates.x * levelHeight == floor(coordinates.x * levelHeight)) {
        coordinates = vec2(coordinates.x + 1.0 / (levelWidth * levelWidth), coordinates.y);
    }
    
    vec2 tilePos = vec2(
        floor(coordinates.x * levelWidth),
        floor(coordinates.y * levelHeight) 
    );

    vec4 tileSpec = texture(levelTexture, coordinates.yx);
    
    float render = round(tileSpec.x * 255);

    if(render != 0) {
        float index = round(255 * (tileSpec.y + 256 * tileSpec.z));

        vec2 texIndex = vec2(
            floor(mod(index, tilesWidth)) / tilesWidth,
            floor(index / tilesWidth) / tilesHeight
        );

        vec2 texOffset = vec2(
            (coordinates.x * levelWidth - tilePos.x) / tilesWidth,
            (coordinates.y * levelHeight - tilePos.y) / tilesHeight
        );
        

        outColor = texture(tilesTexture, texIndex + texOffset);
    } else {
        outColor = vec4(0.0, 0.0, 0.0, 0.0);
    }
}
"""


class ProjectionMatrix:
    """
    A standardized representation of the projection matrix used for the rendering.

    The projection matrix allows to transforms in different ways the rendering. It has to be use for the camera
    projection and the different renderers.

    Attributes
    ----------
    matrix: numpy.ndarray
        The projection matrix in the numpy array format.

    Methods
    -------
    orthographic(right, left, bottom, top)
        Applies a 2D orthographic projection to the matrix.
    scale(scale)
        Scales the projection matrix.
    translate(translation)
        Translates the projection matrix.
    rotate(angle)
        Rotates the projection matrix.
    dot(matrix)
        Composes the projection matrix with another matrix.
    inverse_transform(vector)
        Applies an inverse transformation to a vector.
    """

    def __init__(self, matrix: numpy.ndarray = None):
        """
        Initializes the ProjectionMatrix.

        Parameters
        ----------
        matrix: numpy.ndarray, optional
            The projection matrix in the numpy array format (the identity by default).
        """

        self.matrix = matrix if matrix is not None else numpy.identity(4, dtype=numpy.float32)

    def orthographic(self, right: float, left: float, bottom: float, top: float) -> "ProjectionMatrix":
        """
        Applies a 2D orthographic projection to the matrix.

        This kind of transformation should only be used by the camera by describing its field of view.

        Parameters
        ----------
        right: float
            The extreme right position of the projection.
        left: float
            The extreme left position of the projection.
        bottom: float
            The extreme bottom position of the projection.
        top: float
            The extreme top position of the projection

        Returns
        -------
        projection: ProjectionMatrix
            The current projection matrix with the transformation applied.
        """

        orthographic = numpy.array([
            [2.0 / (right - left), 0.0, 0.0, - (right + left) / (right - left)],
            [0.0, 2.0 / (top - bottom), 0.0, - (top + bottom) / (top - bottom)],
            [0.0, 0.0, - 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=numpy.float32)

        self.matrix = self.matrix.dot(orthographic)

        return self

    def scale(self, scale: [tuple, numpy.ndarray]) -> "ProjectionMatrix":
        """
        Scales the projection matrix.

        Parameters
        ----------
        scale: [tuple, numpy.ndarray]
            The scale factor vector along both directions.

        Returns
        -------
        projection: ProjectionMatrix
            The current projection matrix with the transformation applied.
        """

        scale = array_format(scale, dtype=numpy.float32)

        scale_matrix = numpy.array([
            [scale[0], 0.0, 0.0, 0.0],
            [0.0, scale[1], 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=numpy.float32)

        self.matrix = self.matrix.dot(scale_matrix)

        return self

    def translate(self, translation: [tuple, numpy.ndarray]) -> "ProjectionMatrix":
        """
        Translates the projection matrix.

        Parameters
        ----------
        translation: [tuple, numpy.ndarray]
            The translation vector along both directions.

        Returns
        -------
        projection: ProjectionMatrix
            The current projection matrix with the transformation applied.
        """

        translation = array_format(translation, dtype=numpy.float32)

        translation_matrix = numpy.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [translation[0], translation[1], 0.0, 1.0]
        ], dtype=numpy.float32)

        self.matrix = self.matrix.dot(translation_matrix)

        return self

    def rotate(self, angle: float) -> "ProjectionMatrix":
        """
        Rotates the projection matrix.

        Parameters
        ----------
        angle: float
            The angle by which the matrix is rotated.

        Returns
        -------
        projection: ProjectionMatrix
            The current projection matrix with the transformation applied.
        """

        rotation_matrix = numpy.array([
            [numpy.cos(angle), - numpy.sin(angle), 0.0, 0.0],
            [numpy.sin(angle), numpy.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=numpy.float32)

        self.matrix = self.matrix.dot(rotation_matrix)

        return self

    def dot(self, matrix: numpy.ndarray) -> "ProjectionMatrix":
        """
        Composes the projection matrix with another matrix.

        Parameters
        ----------
        matrix: numpy.ndarray
            The matrix with which the projection will be composed.

        Returns
        -------
        projection: ProjectionMatrix
            The current projection matrix with the transformation applied.
        """

        self.matrix = self.matrix.dot(matrix).astype(numpy.float32)

        return self

    def inverse_transform(self, vector: [tuple, numpy.ndarray]) -> numpy.ndarray:
        """
        Applies an inverse transformation to a vector.

        Parameters
        ----------
        vector: [tuple, numpy.ndarray]
            The vector to transform in the local coordinate system.

        Returns
        -------
        vector: numpy.ndarray
            The transformed vector.
        """

        vector = array_format(vector, dtype=numpy.float32)

        full_vector = numpy.array([[vector[0]], [vector[1]], [1], [0]], dtype=numpy.float32)

        inverse_matrix = numpy.array([
            [1 / self.matrix[0, 0], 0.0, 0.0, 0.0],
            [0.0, 1 / self.matrix[1, 1], 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [- self.matrix[3, 0], - self.matrix[3, 1], 0.0, 1.0]
        ], dtype=numpy.float32)

        transformed = inverse_matrix.dot(full_vector)

        return numpy.array((transformed[0, 0], transformed[1, 0]), dtype=numpy.float32)


class Camera:
    """
    The main rendering camera object.

    The world is rendered through the camera. The field of view specifies the number of "pixels" that can be rendered on
    each axis.

    Attributes
    ----------
    size: numpy.ndarray
        The size of the field of view.
    position: numpy.ndarray
        The position vector of the center point.
    viewport: numpy.ndarray
        The viewport of the camera.
    position_matrix: ProjectionMatrix
        The projection matrix which translates the rendering to the camera position.
    projection_matrix: ProjectionMatrix
        The projection matrix of the 2D orthographic projection.

    Methods
    -------
    transform_world(position)
        Transforms the position vector into the camera coordinate system.
    """

    def __init__(self, size: [tuple, numpy.ndarray]):
        """
        Initializes the Camera.

        Parameters
        ----------
        size: [tuple, numpy.ndarray]
            The size of the field of view.
        """

        self._size = array_format(size, dtype=numpy.float32)
        self._position = array_format((0, 0), dtype=numpy.float32)

        self.viewport = self.size / 2

        self.position_matrix = ProjectionMatrix()

        self.projection_matrix = ProjectionMatrix(). \
            orthographic(- self.viewport[0], self.viewport[0], - self.viewport[1], self.viewport[1])

    @property
    def position(self) -> numpy.ndarray:
        """
        The position property representing the position of the center point of the camera.
        """

        return self._position

    @position.setter
    def position(self, position: [tuple, numpy.ndarray]) -> None:
        """
        Setter function for the position vector.

        Parameters
        ----------
        position: [tuple, numpy.ndarray]
            The position of the center point of the camera.
        """

        self.position_matrix.translate((position[0] - self._position[0], self._position[1] - position[1]))

        self._position = array_format(position, dtype=numpy.float32)

    @property
    def size(self) -> numpy.ndarray:
        """
        The size property representing the width and the height of the field of view.
        """

        return self._size

    @size.setter
    def size(self, size: [tuple, numpy.ndarray]) -> None:
        """
        Setter function for the size vector.

        Parameters
        ----------
        bounds: [tuple, numpy.ndarray]
            The size vector of the field of view.
        """

        self._size = array_format(size, dtype=numpy.float32)
        self.viewport = self._size / 2

        self.projection_matrix = ProjectionMatrix(). \
            orthographic(- self.viewport[0], self.viewport[0], - self.viewport[1], self.viewport[1])

    def transform_world(self, position: numpy.ndarray) -> numpy.ndarray:
        """
        Transforms the position vector into the camera coordinate system.

        Transforms the coordinate of a point expressed in the screen referential to the local camera coordinate system.

        Parameters
        ----------
        position: numpy.ndarray
            The screen relative position vector.

        Returns
        -------
        transformed_position: numpy.ndarray
            The new position vector expressed in the camera coordinate system.
        """

        transform = ProjectionMatrix(self.projection_matrix.matrix).inverse_transform(position)

        transform[0] = self.position[0] - transform[0]
        transform[1] = self.position[1] + transform[1]

        return transform


class Texture:
    """
    A texture OpenGl object.

    This class allows to buffer a numpy array into the OpenGL memory. Note that the image arrays have to be in the RGBA
    unsigned byte format, hence the depth must be 4. When this object is deleted, the OpenGL memory is automatically
    cleared.

    Methods
    -------
    bind(texture)
        Binds the texture to an OpenGL sampler.
    """

    SAMPLER_SPRITE = 0
    SAMPLER_LEVEL = 1
    SAMPLER_TILES = 2

    def __init__(self, context: moderngl.Context, texture: numpy.ndarray, scale: bool = False):
        """
        Initializes the Texture.

        Parameters
        ----------
        context: moderngl.Context
            The main OpenGL context.
        texture: numpy.ndarray
            The image array in a RGBA unsigned byte format.
        scale: bool, optional
            Treats the image as a gray scale if set to True (only used by the level renderer).
        """

        self._width = texture.shape[0]
        self._height = texture.shape[1]

        if scale:
            texture = numpy.flip(texture, axis=1)

            transform = numpy.zeros((self._width, self._height, 3), dtype=numpy.int32)
            transform[:, :, 0] = (texture != 0).astype(numpy.int32) * 255
            transform[:, :, 1] = (texture - 1) % 256
            transform[:, :, 2] = (texture - 1) // 256 % 256

            buffer = transform.reshape(self._width * self._height * 3).astype(numpy.ubyte)

            self._texture = context.texture((self._height, self._width), 3, buffer.tobytes())

        else:
            buffer = texture.reshape(self._width * self._height * 4).astype(numpy.ubyte)

            self._texture = context.texture((self._height, self._width), 4, buffer.tobytes())

        self._texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._texture.repeat_x = False
        self._texture.repeat_y = False

    def __del__(self) -> None:
        """
        Cleans up the GPU memory by releasing the texture buffer.
        """

        self._texture.release()

    def bind(self, sampler: int) -> None:
        """
        Binds the texture to an OpenGL sampler.

        Parameters
        ----------
        sampler: int
            The index of the simple to which the texture will be bound.

        Raises
        ------
        ValueError
            If the sampler index is lesser than 0 or greater than 31.
        """

        if 0 <= sampler < 32:
            self._texture.use(sampler)
        else:
            raise ValueError("The sampler used must be located between 0 and 31.")


class Model:
    """
    A model OpenGL object.

    This object holds the different OpenGL VBOs and VAO used to render a model on screen. When this object is deleted,
    the OpenGL memory is automatically cleared.

    Methods
    -------
    render()
        Renders the model.
    """

    ATTRIBUTE_VERTICES = "vertices"
    ATTRIBUTE_TEXTURES = "textures"

    def __init__(self, context: moderngl.Context, program: moderngl.Program, vertices_buffer: numpy.ndarray,
                 texture_coordinates_buffer: numpy.ndarray, indices_buffer: numpy.ndarray):
        """
        Initializes the Model.

        Parameters
        ----------
        context: moderngl.Context
            The main OpenGL context.
        program: moderngl.Program
            The GLSL shader program used for the rendering.
        vertices_buffer: numpy.ndarray
            The vertex coordinates array.
        texture_coordinates_buffer: numpy.ndarray
            The texture coordinates array.
        indices_buffer: numpy.ndarray
            The texture indices array.
        """

        self._vbo_vertices = context.buffer(vertices_buffer.astype(numpy.float32).tobytes())
        self._vbo_texture_coordinates = context.buffer(texture_coordinates_buffer.astype(numpy.float32).tobytes())
        self._vbo_indices = context.buffer(indices_buffer.astype(numpy.int32).tobytes())

        vao_content = [
            (self._vbo_vertices, '2f', Model.ATTRIBUTE_VERTICES),
            (self._vbo_texture_coordinates, '2f', Model.ATTRIBUTE_TEXTURES)
        ]

        self._vao = context.vertex_array(program, vao_content, self._vbo_indices)

    def __del__(self) -> None:
        """
        Cleans up the GPU memory by releasing the VAO and VBOs.
        """

        self._vao.release()

        self._vbo_vertices.release()
        self._vbo_texture_coordinates.release()
        self._vbo_indices.release()

    def render(self) -> None:
        """
        Renders the model.
        """

        self._vao.render()


class ShaderProgram:
    """
    A program GLSL object.

    The object represents a GLSL shader program. It is used to render the models. When this object is deleted, the
    OpenGL memory is automatically cleared.

    Attributes
    ----------
    program: moderngl.Program
        The GLSL program object.

    Methods
    -------
    set_uniform(name, value)
        Sets the value of a program uniform.
    """

    UNIFORM_POSITION = "position"
    UNIFORM_PROJECTION = "projection"

    UNIFORM_SAMPLER = "sampler"

    UNIFORM_TILES_SAMPLER = "tilesTexture"
    UNIFORM_TILES_WIDTH = "tilesWidth"
    UNIFORM_TILES_HEIGHT = "tilesHeight"

    UNIFORM_LEVEL_SAMPLER = "levelTexture"
    UNIFORM_LEVEL_WIDTH = "levelWidth"
    UNIFORM_LEVEL_HEIGHT = "levelHeight"

    def __init__(self, context: moderngl.Context, source_vertex: str, source_fragment: str):
        """
        Initializes the ShaderProgram.

        Parameters
        ----------
        context: moderngl.Context
            The main OpenGL context.
        source_vertex: str
            The GLSL source code of the vertex shader.
        source_fragment: str
            The GLSL source code of the fragment shader.
        """

        self.program = context.program(vertex_shader=source_vertex, fragment_shader=source_fragment)

    def __del__(self) -> None:
        """
        Cleans up the GPU memory by releasing the program.
        """

        self.program.release()

    def set_uniform(self, name: str, value: any) -> None:
        """
        Sets the value of a program uniform.

        Parameters
        ----------
        name: str
            The name of the program uniform.
        value: any
            The new value of the uniform. Numpy arrays are correctly converted.
        """

        if name in self.program:
            if isinstance(value, numpy.ndarray):
                self.program[name].value = tuple(value.reshape(-1))
            else:
                self.program[name].value = value


class SpriteSet:
    """
    A collection of textures and animation.

    This object contains the textures and the animations used for a sprite.

    Methods
    -------
    register_sprite_texture(texture)
        Registers a new texture.
    register_sprite_animation(period, animation)
        Registers a new animation.
    get_texture(id_animation, pointer)
        Returns the texture to render.
    """

    def __init__(self):
        """
        Initializes the SpriteSet.
        """

        self._textures = []
        self._animations = []

    def register_sprite_texture(self, texture: Texture) -> None:
        """
        Registers a new texture.

        Registers a new texture and gives it the first index available within the sprite set.

        Parameters
        ----------
        texture: Texture
            The texture used for the rendering.
        """

        self._textures.append(texture)

    def register_sprite_animation(self, period: int, animation: tuple) -> None:
        """
        Registers a new animation.

        Registers a new animation and gives it the first index available within the sprite set.

        Parameters
        ----------
        period: int
            The period of the animation expressed in frames (if the animation is static, this should be set to 0).
        animation: tuple of ints
            The indexes of the textures of the animation.
        """

        self._animations.append((period, animation))

    def get_texture(self, id_animation: int, pointer: int) -> (int, Texture):
        """
        Returns the texture to render.

        Returns the texture to render and the next animation pointer. The pointer loop through the whole animation and
        is reset to 0 when the animation is completed.

        Parameters
        ----------
        id_animation: int
            The index of the animation used.
        pointer: int
            The current animation pointer.

        Returns
        -------
        pointer, texture: int, Texture
            The new pointer and the texture to render.
        """

        if self._animations[id_animation][0] == 0 or \
                pointer >= self._animations[id_animation][0] * len(self._animations[id_animation][1]):
            return 0, self._textures[self._animations[id_animation][1][0]]

        texture = self._textures[self._animations[id_animation][1][pointer // self._animations[id_animation][0]]]

        if pointer == self._animations[id_animation][0] * len(self._animations[id_animation][1]) - 1:
            return 0, texture

        return pointer + 1, texture


class TileSet:
    """
    A collection of tile texture concatenated.

    For better performance, the tiles texture are only sampled once. Every tile texture are placed on a single tile set
    texture. Note that no more than 65536 tiles texture can be used.

    Attributes
    ----------
    width: int
        The width of the grid (number of texture per row).
    height: int
        The height of the grid (number of texture per column).

    Methods
    -------
    bind(program)
        Binds the tile set texture.
    """

    def __init__(self, texture: Texture, width: int, height: int):
        """
        Initializes the TileSet.

        Parameters
        ----------
        texture: Texture
            The texture containing the tiles organised in grid.
        width: int
            The width of the grid (number of texture per row).
        height: int
            The height of the grid (number of texture per column).

        Raises
        ------
        ValueError
            If the number of texture exceed 65536 or if the size is invalid.
        """

        self._texture = texture

        if width * height > 65536:
            raise ValueError("The maximum tile index is 65535.")

        if width <= 0 or height <= 0:
            raise ValueError("An tiles must have a width/height of at least 1.")

        self.width = width
        self.height = height

    def bind(self, shader: ShaderProgram) -> None:
        """
        Binds the tile set texture.

        Binds the tile set texture to the tiles sampler and updates the program uniforms related to the size.

        Parameters
        ----------
        shader: ShaderProgram
            The shader program used for the world rendering.
        """

        self._texture.bind(Texture.SAMPLER_TILES)

        shader.set_uniform(ShaderProgram.UNIFORM_TILES_WIDTH, self.width)
        shader.set_uniform(ShaderProgram.UNIFORM_TILES_HEIGHT, self.height)


class ResourceManager(TileManager):
    """
    The graphical resource manager.

    A resource manager must be created to registers the textures used and to render the world. This is only needed if
    the game needs to be rendered. This object should be fully initialized before creating a world. If there is no
    window used for the rendering, this should be created in standalone mode. It contains the main OpenGL context as
    well as the different graphical graphics used for the rendering.

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

    Methods
    -------
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
    """

    BUFFER_QUAD_POSITION = numpy.array([-0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5], dtype=numpy.float64)
    BUFFER_QUAD_TEXTURE_POSITION = numpy.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=numpy.float64)
    BUFFER_QUAD_INDICES = numpy.array([0, 1, 2, 2, 3, 0], dtype=numpy.uint)

    def __init__(self, tile_size: int, scale: float = 1.0, standalone: bool = False, width: int = 800,
                 height: int = 600, glsl_version: int = 330, shader_world: tuple = None, shader_sprite: tuple = None):
        """
        Initializes the ResourceManager.

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
        """

        TileManager.__init__(self, tile_size)

        if not standalone:
            self.context = moderngl.create_context(require=glsl_version)
            self._frame_buffer = self.context.screen
        else:
            self.context = moderngl.create_standalone_context(require=glsl_version)
            self._frame_buffer = self.context.framebuffer([self.context.renderbuffer((width, height))])

        self._frame_buffer.use()

        self.context.clear(1.0, 1.0, 1.0, 1.0)
        self.context.enable(moderngl.BLEND)
        self.context.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.scale = scale

        if not shader_world:
            self.shader_world = ShaderProgram(
                self.context, DEFAULT_SHADER_WORLD_VERTEX, DEFAULT_SHADER_WORLD_FRAGMENT
            )
        else:
            self.shader_world = ShaderProgram(self.context, shader_world[0], shader_world[1])

        self.shader_world.set_uniform(ShaderProgram.UNIFORM_LEVEL_SAMPLER, Texture.SAMPLER_LEVEL)
        self.shader_world.set_uniform(ShaderProgram.UNIFORM_TILES_SAMPLER, Texture.SAMPLER_TILES)

        if not shader_sprite:
            self.shader_sprite = ShaderProgram(
                self.context, DEFAULT_SHADER_SPRITE_VERTEX, DEFAULT_SHADER_SPRITE_FRAGMENT
            )
        else:
            self.shader_sprite = ShaderProgram(self.context, shader_sprite[0], shader_sprite[1])

        self.shader_sprite.set_uniform(ShaderProgram.UNIFORM_SAMPLER, Texture.SAMPLER_SPRITE)

        self.model_world = Model(
            self.context, self.shader_world.program, ResourceManager.BUFFER_QUAD_POSITION,
            ResourceManager.BUFFER_QUAD_TEXTURE_POSITION, ResourceManager.BUFFER_QUAD_INDICES
        )

        self.model_sprite = Model(
            self.context, self.shader_sprite.program, ResourceManager.BUFFER_QUAD_POSITION,
            ResourceManager.BUFFER_QUAD_TEXTURE_POSITION, ResourceManager.BUFFER_QUAD_INDICES
        )

        self.tile_set = None
        self.sprite_sets = {}

        self._background_textures = []
        self._backgrounds = {}

    def __del__(self) -> None:
        """
        Release the current OpenGL context.
        """

        self.context.release()
    
    def register_tile_set(self, texture: numpy.ndarray, width: int, height: int) -> None:
        """
        Registers the main tile set.

        Parameters
        ----------
        texture: numpy.ndarray
            The texture array containing the tiles organised in grid.
        width: int
            The width of the grid (number of texture per row).
        height: int
            The height of the grid (number of texture per column).
        """

        self.tile_set = TileSet(Texture(self.context, texture), width, height)

        self.tile_set.bind(self.shader_world)

    def register_background(self, name: str) -> None:
        """
        Registers a new background.

        Registers a new background under the specified name. A background consists of a bunch of background layers
        stacked on each others.

        Parameters
        ----------
        name: str
            The name of the background.
        """

        self._backgrounds[name] = []

    def register_background_texture(self, texture: numpy.ndarray) -> None:
        """
        Registers a new background texture.

        Registers a new background texture and gives it the first available index.

        Parameters
        ----------
        texture: numpy.ndarray
            The texture array of the background.
        """

        self._background_textures.append(Texture(self.context, texture))

    def register_background_layer(self, name: str, id_texture: int, ratio_x: float, ratio_y: float) -> None:
        """
        Registers a new background layer.

        Registers a new background layer with the specified camera ratios. The ratios indicates how fast the background
        layer moves relatively to the camera speed. A ratio of 0 mean that the background is static and a ratio of 1
        means that the background moves at the same speed as the camera.

        Parameters
        ----------
        name: str
            The name of the background.
        id_texture: int
            The index of the background texture.
        ratio_x: float
            The background over camera speed ratio along the x axis.
        ratio_y: float
            The background over camera speed ratio along the y axis.
        """

        self._backgrounds[name].append((id_texture, ratio_x, ratio_y))

    def register_sprite_set(self, sprite_set: str) -> None:
        """
        Registers a new sprite set.

        Registers a new sprite set under the specified name.

        Parameters
        ----------
        sprite_set: str
            The name of the sprite set.
        """

        self.sprite_sets[sprite_set] = SpriteSet()

    def register_sprite_texture(self, sprite_set: str, texture: numpy.ndarray) -> None:
        """
        Registers a new sprite texture.

        Registers a new sprite texture and gives it the first index available within the sprite set.

        Parameters
        ----------
        sprite_set: str
            The name of the sprite set.
        texture: numpy.ndarray
            The texture array of the sprite.
        """

        self.sprite_sets[sprite_set].register_sprite_texture(Texture(self.context, texture))

    def register_sprite_animation(self, sprite_set: str, period: int, animation: tuple) -> None:
        """
        Registers a new sprite animation.

        Registers a new sprite animation and gives it the first index available within the sprite set.

        Parameters
        ----------
        sprite_set: str
            The name of the sprite set.
        period: int
            The period of the animation expressed in frames (if the animation is static, this should be set to 0).
        animation: tuple of ints
            The indexes of the textures of the animation.
        """

        self.sprite_sets[sprite_set].register_sprite_animation(period, animation)

    def get_last_frame_buffer(self, viewport=None) -> numpy.ndarray:
        """
        Returns the latest rendered frame.

        Parameters
        ----------
        viewport: tuple of int
            The part of the screen to buffer (takes the full frame if set to None).

        Returns
        -------
        frame: numpy.ndarray
            The latest rendered frame.
        """

        if viewport is None:
            viewport = (0, 0) + self._frame_buffer.size

        return numpy.flip(numpy.frombuffer(
            self._frame_buffer.read(viewport=viewport), dtype=numpy.uint8
        ).reshape(
            (viewport[3], viewport[2], 3)), axis=0
        )

    def render_background(self, name: str, camera: Camera) -> None:
        """
        Renders the background.

        Renders the specified background all over the screen, hence this function should be called directly after the
        frame buffer has been cleared and before the rendering of the level and the entities.

        Parameters
        ----------
        name: str
            The name of the background rendered.
        camera: Camera
            The camera used for the rendering.
        """

        for layer in self._backgrounds[name]:
            self._background_textures[layer[0]].bind(0)

            position_delta = ((- camera.position * self.scale * layer[1:]) % camera.size) / camera.size

            for i in range(-1 + (position_delta[0] == 0), 1):
                for j in range(- 1 + (position_delta[1] == 0), 1):
                    position = ProjectionMatrix().translate(position_delta + (i, j)).scale((2, 2))

                    self.shader_sprite.set_uniform(ShaderProgram.UNIFORM_POSITION, position.matrix)
                    self.model_sprite.render()

    def render_sprite(self, renderable: Renderable, camera: Camera) -> None:
        """
        Renders a sprite.

        Renders a renderable sprite considering its animation played and its position in the world.

        Parameters
        ----------
        renderable: Renderable
            The renderable to render.
        camera: Camera
            The camera used for the rendering.
        """

        if renderable.is_transiting():
            animation_pointer, texture = self.sprite_sets[renderable.sprite_set].get_texture(
                renderable.id_animation_transition, renderable.animation_pointer
            )
        else:
            animation_pointer, texture = self.sprite_sets[renderable.sprite_set].get_texture(
                renderable.id_animation, renderable.animation_pointer
            )

        texture.bind(0)

        flip_array = numpy.array((renderable.flip_horizontally, renderable.flip_vertically), dtype=numpy.float32)

        position = ProjectionMatrix().scale(
            renderable.texture_bounds.bounds * (1 - flip_array * 2)
        ).rotate(
            renderable.angle
        ).translate(
            (renderable.position + renderable.texture_bounds.bounds / 2 + renderable.texture_bounds.position)
        ).scale(
            (-1, 1)
        ).dot(
            camera.position_matrix.matrix
        ).scale(
            (self.scale, self.scale)
        )

        self.shader_sprite.set_uniform(ShaderProgram.UNIFORM_POSITION, position.matrix)
        self.model_sprite.render()

        renderable.animation_pointer = animation_pointer

    def increment_animation_pointer(self, renderable: Renderable) -> None:
        """
        Increments the animation pointer of the renderable.

        Increments the animation pointer of the specified renderable. Note that when a renderable is displayed, its
        animation pointer is automatically displayed, this function should be used to keep the animation of non rendered
        object (such as out of camera view objects) synchronized with the others.

        Parameters
        ----------
        renderable: Renderable
            The renderable of which the animation pointer will be incremented.
        """

        if renderable.is_transiting():
            animation_pointer, _ = self.sprite_sets[renderable.sprite_set].get_texture(
                renderable.id_animation_transition, renderable.animation_pointer
            )
        else:
            animation_pointer, _ = self.sprite_sets[renderable.sprite_set].get_texture(
                renderable.id_animation, renderable.animation_pointer
            )

        renderable.animation_pointer = animation_pointer


class LevelRenderer:
    """
    The renderer used for level rendering.

    This class stores a texture of the tiles of the level. This allow to render the level faster than rendering the
    tiles individually. Each time the level is modified, the texture of the level should be updated.

    Methods
    -------
    render(camera)
        Renders the level.
    update_tiles(tiles)
        Updates the tile level array.
    """

    def __init__(self, resources: ResourceManager, tiles: numpy.ndarray, scale: int):
        """
        Initializes the LevelRenderer.

        Parameters
        ----------
        resources: ResourceManager
            The main resource manager used for the rendering.
        tiles: numpy.ndarray
            The tiles array of the level.
        scale: int
            The size of the tiles expressed in distance units.
        """

        self._resources = resources
        self._tiles = tiles.copy()
        self._scale = scale

        self._width = tiles.shape[0]
        self._height = tiles.shape[1]

        self._texture = Texture(resources.context, self._tiles, scale=True)

        self._bind()

    def _bind(self) -> None:
        """
        Binds the level into the GPU memory.
        """

        self._texture.bind(Texture.SAMPLER_LEVEL)

        self._resources.shader_world.set_uniform(ShaderProgram.UNIFORM_LEVEL_WIDTH, self._width)
        self._resources.shader_world.set_uniform(ShaderProgram.UNIFORM_LEVEL_HEIGHT, self._height)

    def render(self, camera: Camera) -> None:
        """
        Renders the level.

        Parameters
        ----------
        camera: Camera
            The camera used for the rendering.
        """

        position = ProjectionMatrix().translate(
            (0.5, 0.5)
        ).scale(
            (- self._scale * self._width, self._scale * self._height)
        ).dot(
            camera.position_matrix.matrix
        ).scale(
            (self._resources.scale, self._resources.scale)
        )

        self._resources.shader_world.set_uniform(ShaderProgram.UNIFORM_POSITION, position.matrix)
        self._resources.shader_world.set_uniform(ShaderProgram.UNIFORM_PROJECTION, camera.projection_matrix.matrix)

        self._resources.model_world.render()

    def update_tiles(self, tiles: numpy.ndarray):
        """
        Updates the tile level array.

        Updates the local copy of the level array. This will update the OpenGL object only if change occurred.

        Parameters
        ----------
        tiles: numpy.ndarray
            The tiles array of the level.
        """

        if not numpy.array_equal(self._tiles, tiles):
            self._tiles = tiles.copy()

            self._texture = Texture(self._resources.context, self._tiles, scale=True)

            if self._tiles.shape[0] != self._width or self._tiles.shape[1] != self._height:
                self._width = tiles.shape[0]
                self._height = tiles.shape[1]

                self._bind()
            else:
                self._texture.bind(Texture.SAMPLER_LEVEL)


class WorldRenderer:
    """
    The renderer used for world rendering.

    This class allows to render the level and the entities of the world. It relies on a level renderer. The entities out
    of the camera field are not displayed for better performance.

    Methods
    -------
    render(camera)
        Renders the world.
    """

    def __init__(self, resources: ResourceManager, world: World):
        """
        Initializes the WorldRenderer.

        Parameters
        ----------
        resources: ResourceManager
            The main resource manager used for the rendering.
        world: World
            The main world to render.
        """

        self._resources = resources
        self._world = world

        self._level_renderer = LevelRenderer(resources, self._world.tiles, self._resources.tile_size)

    def render(self, camera: Camera) -> None:
        """
        Renders the world.

        Renders the world by rendering the following in order: background, level and entities. Note that the entities
        out of the screen are not rendered for better performances.

        Parameters
        ----------
        camera: Camera
            The camera used for the rendering.
        """

        self._resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_PROJECTION, ProjectionMatrix().matrix)

        if self._world.background is not None:
            self._resources.render_background(self._world.background, camera)

        self._level_renderer.update_tiles(self._world.tiles)
        self._level_renderer.render(camera)

        self._resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_PROJECTION, camera.projection_matrix.matrix)

        for world_object in self._world.world_objects:
            if isinstance(world_object, Renderable):
                if not world_object.should_be_destroyed and world_object.visible:
                    if numpy.logical_and(*(
                            (numpy.abs(world_object.position - camera.position) -
                             world_object.texture_bounds.bounds) * self._resources.scale < camera.viewport)):
                        self._resources.render_sprite(world_object, camera)
                    else:
                        self._resources.increment_animation_pointer(world_object)


class RenderLoop(LogicLoop):
    """
    The main game loop function with addition of rendering routine.

    This object is callable, hence is should be called as if it was a function. When called, the loop starts and run the
    logic function at each tick, at a specified tick rate and run the render function at each frame, at a specified
    frame rate.

    Methods
    -------
    stop()
        Stops the main loop.
    """

    def __init__(self, tick_per_second: float, frame_per_second: float, function_logic: callable,
                 function_render: callable, function_error: callable):
        """
        Initializes the RenderLoop.

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
        function_error: callable
            The error function called whenever an exception occurs within the loop.
        """

        LogicLoop.__init__(self, tick_per_second, function_logic, function_error)

        self._function_render = function_render
        self._frame_period = 1.0 / frame_per_second

        self._frame = 0

    def __call__(self, number_of_ticks: int = None, max_speed: bool = False) -> None:
        """
        Runs the main logic function.

        Parameters
        ----------
        number_of_ticks: int, optional
            The number of ticks over which the logic will be performed. Il this parameter is not set, the loop will run
            until the stop function is called.
        max_speed: bool, optional
            Ignores the tick rate and runs at maximum speed if set to True.
        """

        target_tick = self._tick + number_of_ticks if number_of_ticks else -1

        current_time = LogicLoop._get_current_time()
        reference_logic = current_time
        reference_render = current_time

        self._running = True

        try:
            if max_speed:
                while self._running and self._tick != target_tick:
                    self._do_logic()

                    if LogicLoop._get_current_time() - reference_render >= self._frame_period:
                        self._do_render()

                        reference_render += self._frame_period

            else:
                while self._running and self._tick != target_tick:
                    if max_speed or LogicLoop._get_current_time() - reference_logic >= self._tick_period:
                        self._do_logic()

                        reference_logic += self._tick_period

                    if LogicLoop._get_current_time() - reference_render >= self._frame_period:
                        self._do_render()

                        reference_render += self._frame_period

                    sleep_time = min(self._tick_period + reference_logic, self._frame_period + reference_render) - \
                        LogicLoop._get_current_time()

                    if sleep_time > 0:
                        sleep(sleep_time)
        except BaseException:
            self._function_error()

            raise

        self._running = False

    def _do_render(self) -> None:
        """
        Calls the render routine.
        """

        self._function_render(self._frame)

        self._frame += 1
