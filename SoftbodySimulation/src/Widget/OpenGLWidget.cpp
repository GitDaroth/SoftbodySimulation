#include "Widget/OpenGLWidget.h"

OpenGLWidget::OpenGLWidget(QWidget *parent) : 
	QOpenGLWidget(parent),
	m_camera(std::make_shared<Camera>()),
	m_isParticleRenderingEnabled(false),
	m_isShapeMeshRenderingEnabled(true),
	m_particleVertexBuffer(QOpenGLBuffer::VertexBuffer),
	m_shapeMeshVertexBuffer(QOpenGLBuffer::VertexBuffer),
	m_shapeMeshIndexBuffer(QOpenGLBuffer::IndexBuffer),
	m_skyboxVertexBuffer(QOpenGLBuffer::VertexBuffer),
	m_skyboxTexture(QOpenGLTexture::TargetCubeMap),
	m_collisionBoxesBoundaryVertexBuffer(QOpenGLBuffer::VertexBuffer),
	m_collisionBoxesObjectVertexBuffer(QOpenGLBuffer::VertexBuffer)
{
	setFormat(QSurfaceFormat::defaultFormat());
}

OpenGLWidget::~OpenGLWidget()
{
	makeCurrent();

	m_particleVertexBuffer.destroy();
	m_particleVertexArrayObject.destroy();

	m_shapeMeshVertexBuffer.destroy();
	m_shapeMeshIndexBuffer.destroy();
	m_shapeMeshVertexArrayObject.destroy();

	m_skyboxVertexBuffer.destroy();
	m_skyboxVertexArrayObject.destroy();

	m_collisionBoxesBoundaryVertexBuffer.destroy();
	m_collisionBoxesBoundaryVertexArrayObject.destroy();

	m_collisionBoxesObjectVertexBuffer.destroy();
	m_collisionBoxesObjectVertexArrayObject.destroy();

	doneCurrent();
}

std::shared_ptr<Camera> OpenGLWidget::getCamera()
{
	return m_camera;
}

void OpenGLWidget::setPBDSolver(std::shared_ptr<PBDSolver> pbdSolver)
{
	m_pbdSolver = pbdSolver;
}

void OpenGLWidget::enableParticleRendering(bool isParticleRenderingEnabled)
{
	m_isParticleRenderingEnabled = isParticleRenderingEnabled;
}

void OpenGLWidget::enableShapeMeshRendering(bool isShapeMeshRenderingEnabled)
{
	m_isShapeMeshRenderingEnabled = isShapeMeshRenderingEnabled;
}

void OpenGLWidget::initializeGL()
{
	initializeOpenGLFunctions();

	prepareParticles();
	prepareShapeMeshes();
	prepareCollisionBoxes();
	prepareSkybox();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);
	glClearColor(0.2f, 0.2f, 0.2f, 1.f);
}

void OpenGLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	if(m_isParticleRenderingEnabled)
		renderParticles();
	if(m_isShapeMeshRenderingEnabled)
		renderShapeMeshes();
	renderCollisionBoxes();
	renderSkybox();
}

void OpenGLWidget::resizeGL(int width, int height)
{
	m_camera->setAspectRatio(width / (float)height);
}

void OpenGLWidget::prepareParticles()
{
	m_particleShaderProgram = std::make_shared<QOpenGLShaderProgram>();
	m_particleShaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "assets/shaders/sphereImpostor.vert");
	m_particleShaderProgram->addShaderFromSourceFile(QOpenGLShader::Geometry, "assets/shaders/sphereImpostor.geom");
	m_particleShaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "assets/shaders/sphereImpostor.frag");
	m_particleShaderProgram->link();
	m_particleShaderProgram->bind();

	m_particleVertexArrayObject.create();
	m_particleVertexArrayObject.bind();

	m_particleVertexBuffer.create();
	m_particleVertexBuffer.bind();
	m_particleVertexBuffer.setUsagePattern(QOpenGLBuffer::StreamDraw);

	m_particleShaderProgram->enableAttributeArray("position");
	// attribLocation, type, offset, tupleSize, stride in bytes
	m_particleShaderProgram->setAttributeBuffer("position", GL_FLOAT, 0, 3, 3 * sizeof(GLfloat));

	m_particleVertexBuffer.release();
	m_particleVertexArrayObject.release();
	m_particleShaderProgram->release();
}

void OpenGLWidget::renderParticles()
{
	int particleCount = m_pbdSolver->getParticles().size();
	if (particleCount <= 0)
		return;

	m_particleShaderProgram->bind();
	m_particleShaderProgram->setUniformValue("color", QVector3D(0.2f, 0.6f, 0.2f));
	m_particleShaderProgram->setUniformValue("viewMatrix", m_camera->getViewMatrix());
	m_particleShaderProgram->setUniformValue("projectionMatrix", m_camera->getProjectionMatrix());

	m_particleVertexArrayObject.bind();

	m_particleVertexBuffer.bind();
	m_particleVertexBuffer.allocate(m_pbdSolver->getParticlePositions(), particleCount * 3 * sizeof(float));
	m_particleVertexBuffer.release();

	glDrawArrays(GL_POINTS, 0, particleCount);

	m_skyboxTexture.release();
	m_particleVertexArrayObject.release();

	m_particleShaderProgram->release();
}

void OpenGLWidget::prepareSkybox()
{
	m_skyboxShaderProgram = std::make_shared<QOpenGLShaderProgram>();
	m_skyboxShaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "assets/shaders/skybox.vert");
	m_skyboxShaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "assets/shaders/skybox.frag");
	m_skyboxShaderProgram->link();
	m_skyboxShaderProgram->bind();

	m_skyboxVertexArrayObject.create();
	m_skyboxVertexArrayObject.bind();

	float vertices[] =
	{
		-1.0f, +1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		+1.0f, -1.0f, -1.0f,
		+1.0f, -1.0f, -1.0f,
		+1.0f, +1.0f, -1.0f,
		-1.0f, +1.0f, -1.0f,

		-1.0f, -1.0f, +1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, +1.0f, -1.0f,
		-1.0f, +1.0f, -1.0f,
		-1.0f, +1.0f, +1.0f,
		-1.0f, -1.0f, +1.0f,

		+1.0f, -1.0f, -1.0f,
		+1.0f, -1.0f, +1.0f,
		+1.0f, +1.0f, +1.0f,
		+1.0f, +1.0f, +1.0f,
		+1.0f, +1.0f, -1.0f,
		+1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f, +1.0f,
		-1.0f, +1.0f, +1.0f,
		+1.0f, +1.0f, +1.0f,
		+1.0f, +1.0f, +1.0f,
		+1.0f, -1.0f, +1.0f,
		-1.0f, -1.0f, +1.0f,

		-1.0f, +1.0f, -1.0f,
		+1.0f, +1.0f, -1.0f,
		+1.0f, +1.0f, +1.0f,
		+1.0f, +1.0f, +1.0f,
		-1.0f, +1.0f, +1.0f,
		-1.0f, +1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, +1.0f,
		+1.0f, -1.0f, -1.0f,
		+1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, +1.0f,
		+1.0f, -1.0f, +1.0f
	};

	m_skyboxVertexBuffer.create();
	m_skyboxVertexBuffer.bind();
	m_skyboxVertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
	m_skyboxVertexBuffer.allocate(vertices, 36 * 3 * sizeof(GLfloat));

	m_skyboxShaderProgram->enableAttributeArray("position");
	// attribLocation, type, offset, tupleSize, stride in bytes
	m_skyboxShaderProgram->setAttributeBuffer("position", GL_FLOAT, 0, 3, 3 * sizeof(GLfloat));

	m_skyboxVertexBuffer.release();
	m_skyboxVertexArrayObject.release();

	const QImage posx = QImage("assets/textures/skybox/posx.jpg").convertToFormat(QImage::Format_RGBA8888);
	const QImage negx = QImage("assets/textures/skybox/negx.jpg").convertToFormat(QImage::Format_RGBA8888);
	const QImage posy = QImage("assets/textures/skybox/posy.jpg").convertToFormat(QImage::Format_RGBA8888);
	const QImage negy = QImage("assets/textures/skybox/negy.jpg").convertToFormat(QImage::Format_RGBA8888);
	const QImage posz = QImage("assets/textures/skybox/posz.jpg").convertToFormat(QImage::Format_RGBA8888);
	const QImage negz = QImage("assets/textures/skybox/negz.jpg").convertToFormat(QImage::Format_RGBA8888);

	m_skyboxTexture.create();
	m_skyboxTexture.setSize(posx.width(), posx.height(), posx.depth());
	m_skyboxTexture.setFormat(QOpenGLTexture::RGBA8_UNorm);
	m_skyboxTexture.allocateStorage();

	m_skyboxTexture.setData(0, 0, QOpenGLTexture::CubeMapPositiveX, QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, posx.constBits(), Q_NULLPTR);
	m_skyboxTexture.setData(0, 0, QOpenGLTexture::CubeMapPositiveY, QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, posy.constBits(), Q_NULLPTR);
	m_skyboxTexture.setData(0, 0, QOpenGLTexture::CubeMapPositiveZ, QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, posz.constBits(), Q_NULLPTR);
	m_skyboxTexture.setData(0, 0, QOpenGLTexture::CubeMapNegativeX, QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, negx.constBits(), Q_NULLPTR);
	m_skyboxTexture.setData(0, 0, QOpenGLTexture::CubeMapNegativeY, QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, negy.constBits(), Q_NULLPTR);
	m_skyboxTexture.setData(0, 0, QOpenGLTexture::CubeMapNegativeZ, QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, negz.constBits(), Q_NULLPTR);

	m_skyboxTexture.setWrapMode(QOpenGLTexture::ClampToEdge);
	m_skyboxTexture.setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
	m_skyboxTexture.setMagnificationFilter(QOpenGLTexture::LinearMipMapLinear);

	m_skyboxShaderProgram->release();
}

void OpenGLWidget::renderSkybox()
{
	glDepthFunc(GL_LEQUAL);

	m_skyboxShaderProgram->bind();

	// remove translational part
	QMatrix4x4 viewMatrix = m_camera->getViewMatrix();
	viewMatrix.setColumn(3, QVector4D(0.f, 0.f, 0.f, 1.f));

	m_skyboxShaderProgram->setUniformValue("viewMatrix", viewMatrix);
	m_skyboxShaderProgram->setUniformValue("projectionMatrix", m_camera->getProjectionMatrix());

	m_skyboxVertexArrayObject.bind();

	glActiveTexture(GL_TEXTURE0);
	m_skyboxShaderProgram->setUniformValue("skybox", 0);
	m_skyboxTexture.bind();

	glDrawArrays(GL_TRIANGLES, 0, 36);

	m_skyboxTexture.release();
	m_skyboxVertexArrayObject.release();

	m_skyboxShaderProgram->release();

	glDepthFunc(GL_LESS);
}

void OpenGLWidget::prepareShapeMeshes()
{
	m_shapeMeshShaderProgram = std::make_shared<QOpenGLShaderProgram>();
	m_shapeMeshShaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "assets/shaders/shapeMesh.vert");
	m_shapeMeshShaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "assets/shaders/shapeMesh.frag");
	m_shapeMeshShaderProgram->link();
	m_shapeMeshShaderProgram->bind();

	m_shapeMeshVertexArrayObject.create();
	m_shapeMeshVertexArrayObject.bind();

	m_shapeMeshVertexBuffer.create();
	m_shapeMeshVertexBuffer.bind();
	m_shapeMeshVertexBuffer.setUsagePattern(QOpenGLBuffer::StreamDraw);

	m_shapeMeshShaderProgram->enableAttributeArray("position");
	// attribLocation, type, offset, tupleSize, stride in bytes
	m_shapeMeshShaderProgram->setAttributeBuffer("position", GL_FLOAT, 0, 3, 6 * sizeof(GLfloat));

	m_shapeMeshShaderProgram->enableAttributeArray("normal");
	m_shapeMeshShaderProgram->setAttributeBuffer("normal", GL_FLOAT, 3 * sizeof(GLfloat), 3, 6 * sizeof(GLfloat));

	m_shapeMeshVertexBuffer.release();
	m_shapeMeshVertexArrayObject.release();

	m_shapeMeshIndexBuffer.create();
	m_shapeMeshIndexBuffer.bind();
	m_shapeMeshIndexBuffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
	m_shapeMeshIndexBuffer.release();

	m_shapeMeshShaderProgram->release();
}

void OpenGLWidget::renderShapeMeshes()
{
	int vertexCount = m_pbdSolver->getVertexCount();
	int vertexIndexCount = m_pbdSolver->getVertexIndexCount();
	if (vertexCount <= 0 || vertexIndexCount <= 0)
		return;

	m_shapeMeshShaderProgram->bind();
	QMatrix4x4 modelMatrix;
	modelMatrix.setToIdentity();
	m_shapeMeshShaderProgram->setUniformValue("usePhongShading", true);
	m_shapeMeshShaderProgram->setUniformValue("color", QVector3D(0.2f, 0.6f, 0.2f));
	m_shapeMeshShaderProgram->setUniformValue("modelMatrix", modelMatrix);
	m_shapeMeshShaderProgram->setUniformValue("viewMatrix", m_camera->getViewMatrix());
	m_shapeMeshShaderProgram->setUniformValue("projectionMatrix", m_camera->getProjectionMatrix());

	m_shapeMeshVertexArrayObject.bind();

	m_shapeMeshVertexBuffer.bind();
	m_shapeMeshVertexBuffer.allocate(m_pbdSolver->getVertices(), 6 * vertexCount * sizeof(float));
	m_shapeMeshVertexBuffer.release();

	m_shapeMeshIndexBuffer.bind();
	m_shapeMeshIndexBuffer.allocate(m_pbdSolver->getVertexIndices(), vertexIndexCount * sizeof(unsigned int));

	glDrawElements(GL_TRIANGLES, vertexIndexCount, GL_UNSIGNED_INT, 0);

	m_shapeMeshIndexBuffer.release();
	m_shapeMeshVertexArrayObject.release();

	m_shapeMeshShaderProgram->release();
}

void OpenGLWidget::prepareCollisionBoxes()
{
	m_collisionBoxesShaderProgram = std::make_shared<QOpenGLShaderProgram>();
	m_collisionBoxesShaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "assets/shaders/shapeMesh.vert");
	m_collisionBoxesShaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "assets/shaders/shapeMesh.frag");
	m_collisionBoxesShaderProgram->link();
	m_collisionBoxesShaderProgram->bind();

	m_collisionBoxesBoundaryVertexArrayObject.create();
	m_collisionBoxesBoundaryVertexArrayObject.bind();

	float boundaryVertices[] =
	{
		// Front Quad
		-1.f, 1.f, 1.f,	// position
		-1.f, 1.f, 1.f,	// normal
		-1.f, -1.f, 1.f,
		-1.f, -1.f, 1.f,

		-1.f, -1.f, 1.f,
		-1.f, -1.f, 1.f,
		1.f, -1.f, 1.f,
		1.f, -1.f, 1.f,

		1.f, -1.f, 1.f,
		1.f, -1.f, 1.f,
		1.f, 1.f, 1.f,
		1.f, 1.f, 1.f,

		1.f, 1.f, 1.f,
		1.f, 1.f, 1.f,
		-1.f, 1.f, 1.f,
		-1.f, 1.f, 1.f,

		// Back Quad
		1.f, 1.f, -1.f,
		1.f, 1.f, -1.f,
		1.f, -1.f, -1.f,
		1.f, -1.f, -1.f,

		1.f, -1.f, -1.f,
		1.f, -1.f, -1.f,
		-1.f, -1.f, -1.f,
		-1.f, -1.f, -1.f,

		-1.f, -1.f, -1.f,
		-1.f, -1.f, -1.f,
		-1.f, 1.f, -1.f,
		-1.f, 1.f, -1.f,

		-1.f, 1.f, -1.f,
		-1.f, 1.f, -1.f,
		1.f, 1.f, -1.f,
		1.f, 1.f, -1.f,

		// Connections Front -> Back
		-1.f, 1.f, 1.f,
		-1.f, 1.f, 1.f,
		-1.f, 1.f, -1.f,
		-1.f, 1.f, -1.f,

		-1.f, -1.f, 1.f,
		-1.f, -1.f, 1.f,
		-1.f, -1.f, -1.f,
		-1.f, -1.f, -1.f,

		1.f, -1.f, 1.f,
		1.f, -1.f, 1.f,
		1.f, -1.f, -1.f,
		1.f, -1.f, -1.f,

		1.f, 1.f, 1.f,
		1.f, 1.f, 1.f,
		1.f, 1.f, -1.f,
		1.f, 1.f, -1.f
	};

	m_collisionBoxesBoundaryVertexBuffer.create();
	m_collisionBoxesBoundaryVertexBuffer.bind();
	m_collisionBoxesBoundaryVertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
	m_collisionBoxesBoundaryVertexBuffer.allocate(boundaryVertices, 24 * 6 * sizeof(GLfloat));

	m_collisionBoxesShaderProgram->enableAttributeArray("position");
	m_collisionBoxesShaderProgram->setAttributeBuffer("position", GL_FLOAT, 0, 3, 6 * sizeof(GLfloat));
	m_collisionBoxesShaderProgram->enableAttributeArray("normal");
	m_collisionBoxesShaderProgram->setAttributeBuffer("normal", GL_FLOAT, 3 * sizeof(GLfloat), 3, 6 * sizeof(GLfloat));

	m_collisionBoxesBoundaryVertexBuffer.release();
	m_collisionBoxesBoundaryVertexArrayObject.release();


	m_collisionBoxesObjectVertexArrayObject.create();
	m_collisionBoxesObjectVertexArrayObject.bind();

	float objectVertices[] =
	{
		// Front Quad
		-1.f, 1.f, 1.f,	// position
		0.f, 0.f, 1.f,	// normal
		-1.f, -1.f, 1.f,
		0.f, 0.f, 1.f,
		1.f, 1.f, 1.f,
		0.f, 0.f, 1.f,

		1.f, 1.f, 1.f,
		0.f, 0.f, 1.f,
		-1.f, -1.f, 1.f,
		0.f, 0.f, 1.f,
		1.f, -1.f, 1.f,
		0.f, 0.f, 1.f,

		// Back Quad
		1.f, 1.f, -1.f,
		0.f, 0.f, -1.f,
		1.f, -1.f, -1.f,
		0.f, 0.f, -1.f,
		-1.f, 1.f, -1.f,
		0.f, 0.f, -1.f,

		-1.f, 1.f, -1.f,
		0.f, 0.f, -1.f,
		1.f, -1.f, -1.f,
		0.f, 0.f, -1.f,
		-1.f, -1.f, -1.f,
		0.f, 0.f, -1.f,

		// Top Quad
		-1.f, 1.f, -1.f,
		0.f, 1.f, 0.f,
		-1.f, 1.f, 1.f,
		0.f, 1.f, 0.f,
		1.f, 1.f, -1.f,
		0.f, 1.f, 0.f,

		1.f, 1.f, -1.f,
		0.f, 1.f, 0.f,
		-1.f, 1.f, 1.f,
		0.f, 1.f, 0.f,
		1.f, 1.f, 1.f,
		0.f, 1.f, 0.f,

		// Bottom Quad
		1.f, -1.f, -1.f,
		0.f, -1.f, 0.f,
		1.f, -1.f, 1.f,
		0.f, -1.f, 0.f,
		-1.f, -1.f, -1.f,
		0.f, -1.f, 0.f,

		-1.f, -1.f, -1.f,
		0.f, -1.f, 0.f,
		1.f, -1.f, 1.f,
		0.f, -1.f, 0.f,
		-1.f, -1.f, 1.f,
		0.f, -1.f, 0.f,

		// Right Quad
		1.f, 1.f, 1.f,
		1.f, 0.f, 0.f,
		1.f, -1.f, 1.f,
		1.f, 0.f, 0.f,
		1.f, 1.f, -1.f,
		1.f, 0.f, 0.f,

		1.f, 1.f, -1.f,
		1.f, 0.f, 0.f,
		1.f, -1.f, 1.f,
		1.f, 0.f, 0.f,
		1.f, -1.f, -1.f,
		1.f, 0.f, 0.f,

		// Left Quad
		-1.f, 1.f, -1.f,
		-1.f, 0.f, 0.f,
		-1.f, -1.f, -1.f,
		-1.f, 0.f, 0.f,
		-1.f, 1.f, 1.f,
		-1.f, 0.f, 0.f,

		-1.f, 1.f, 1.f,
		-1.f, 0.f, 0.f,
		-1.f, -1.f, -1.f,
		-1.f, 0.f, 0.f,
		-1.f, -1.f, 1.f,
		-1.f, 0.f, 0.f
	};

	m_collisionBoxesObjectVertexBuffer.create();
	m_collisionBoxesObjectVertexBuffer.bind();
	m_collisionBoxesObjectVertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
	m_collisionBoxesObjectVertexBuffer.allocate(objectVertices, 36 * 6 * sizeof(GLfloat));

	m_collisionBoxesShaderProgram->enableAttributeArray("position");
	m_collisionBoxesShaderProgram->setAttributeBuffer("position", GL_FLOAT, 0, 3, 6 * sizeof(GLfloat));
	m_collisionBoxesShaderProgram->enableAttributeArray("normal");
	m_collisionBoxesShaderProgram->setAttributeBuffer("normal", GL_FLOAT, 3 * sizeof(GLfloat), 3, 6 * sizeof(GLfloat));

	m_collisionBoxesObjectVertexBuffer.release();
	m_collisionBoxesObjectVertexArrayObject.release();

	m_collisionBoxesShaderProgram->release();
}

void OpenGLWidget::renderCollisionBoxes()
{
	m_collisionBoxesShaderProgram->bind();

	m_collisionBoxesShaderProgram->setUniformValue("viewMatrix", m_camera->getViewMatrix());
	m_collisionBoxesShaderProgram->setUniformValue("projectionMatrix", m_camera->getProjectionMatrix());

	QVector<std::shared_ptr<BoxConstraint>> boxes = m_pbdSolver->getBoxConstraints();
	
	for (auto box : boxes)
	{
		QMatrix4x4 modelMatrix;
		modelMatrix.setToIdentity();
		modelMatrix.translate(box->getPosition());
		modelMatrix.rotate(box->getAngle(), box->getRotationAxis());
		modelMatrix.scale(box->getHalfDimension());

		m_collisionBoxesShaderProgram->setUniformValue("modelMatrix", modelMatrix);

		if (box->getIsBoundary())
		{
			m_collisionBoxesShaderProgram->setUniformValue("usePhongShading", false);
			m_collisionBoxesShaderProgram->setUniformValue("color", QVector3D(0.8f, 0.8f, 0.8f));
			m_collisionBoxesBoundaryVertexArrayObject.bind();
			glDrawArrays(GL_LINES, 0, 24);
			m_collisionBoxesBoundaryVertexArrayObject.release();
		}
		else
		{
			m_collisionBoxesShaderProgram->setUniformValue("usePhongShading", true);
			m_collisionBoxesShaderProgram->setUniformValue("color", QVector3D(0.2f, 0.2f, 0.2f));
			m_collisionBoxesObjectVertexArrayObject.bind();
			glDrawArrays(GL_TRIANGLES, 0, 36);
			m_collisionBoxesObjectVertexArrayObject.release();

			m_collisionBoxesShaderProgram->setUniformValue("usePhongShading", false);
			m_collisionBoxesShaderProgram->setUniformValue("color", QVector3D(0.8f, 0.8f, 0.8f));
			m_collisionBoxesBoundaryVertexArrayObject.bind();
			glDrawArrays(GL_LINES, 0, 24);
			m_collisionBoxesBoundaryVertexArrayObject.release();
		}
	}

	m_collisionBoxesShaderProgram->release();
}
