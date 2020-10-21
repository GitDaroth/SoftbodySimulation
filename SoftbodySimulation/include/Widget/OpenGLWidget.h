#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLShader>
#include <QOpenGLTexture>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include "Camera/Camera.h"
#include <PBDSolver.h>

class OpenGLWidget : public QOpenGLWidget, public QOpenGLFunctions_3_3_Core
{
	Q_OBJECT

public:
	OpenGLWidget(QWidget *parent);
	~OpenGLWidget();

	std::shared_ptr<Camera> getCamera();
	void setPBDSolver(std::shared_ptr<PBDSolver> pbdSolver);

	void enableParticleRendering(bool isParticleRenderingEnabled);
	void enableShapeMeshRendering(bool isShapeMeshRenderingEnabled);

protected:
	virtual void initializeGL() override;
	virtual void paintGL() override;
	virtual void resizeGL(int width, int height) override;

private:
	void prepareParticles();
	void renderParticles();
	void prepareSkybox();
	void renderSkybox();
	void prepareShapeMeshes();
	void renderShapeMeshes();
	void prepareCollisionBoxes();
	void renderCollisionBoxes();

	std::shared_ptr<Camera> m_camera;
	std::shared_ptr<PBDSolver> m_pbdSolver;

	bool m_isParticleRenderingEnabled;
	std::shared_ptr<QOpenGLShaderProgram> m_particleShaderProgram;
	QOpenGLVertexArrayObject m_particleVertexArrayObject;
	QOpenGLBuffer m_particleVertexBuffer;

	bool m_isShapeMeshRenderingEnabled;
	std::shared_ptr<QOpenGLShaderProgram> m_shapeMeshShaderProgram;
	QOpenGLVertexArrayObject m_shapeMeshVertexArrayObject;
	QOpenGLBuffer m_shapeMeshVertexBuffer;
	QOpenGLBuffer m_shapeMeshIndexBuffer;

	std::shared_ptr<QOpenGLShaderProgram> m_skyboxShaderProgram;
	QOpenGLVertexArrayObject m_skyboxVertexArrayObject;
	QOpenGLBuffer m_skyboxVertexBuffer;
	QOpenGLTexture m_skyboxTexture;

	std::shared_ptr<QOpenGLShaderProgram> m_collisionBoxesShaderProgram;
	QOpenGLVertexArrayObject m_collisionBoxesBoundaryVertexArrayObject;
	QOpenGLVertexArrayObject m_collisionBoxesObjectVertexArrayObject;
	QOpenGLBuffer m_collisionBoxesBoundaryVertexBuffer;
	QOpenGLBuffer m_collisionBoxesObjectVertexBuffer;
};
