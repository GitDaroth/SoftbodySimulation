#include "Widget/MainWindow.h"
#include <QResizeEvent>
#include <QTimer>
#include <QThread>

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	m_fpsAveragingTime(0.5f),
	m_fpsFrameCount(0),
	m_fpsAccumulatedTime(0.f),
	m_elapsedSimulationTime(0.f),
	m_targetTimestep(1.f / 60.f),
	m_elapsedTimestep(0.f),
	m_hasSimulationStarted(false),
	m_wasScenarioReset(true),
	m_currentScenario(nullptr)
{
	m_ui.setupUi(this);

	m_pbdSolver = std::make_shared<PBDSolver>();

	m_ui.openGLWidget->setPBDSolver(m_pbdSolver);

	m_controlPanel = new ControlPanel();
	m_controlPanel->setParent(m_ui.openGLWidget);

	connect(m_controlPanel, &ControlPanel::startSimulationEvent, this, &MainWindow::onStartSimulationEvent);
	connect(m_controlPanel, &ControlPanel::pauseSimulationEvent, this, &MainWindow::onPauseSimulationEvent);
	connect(m_controlPanel, &ControlPanel::stepSimulationEvent, this, &MainWindow::onStepSimulationEvent);
	connect(m_controlPanel, &ControlPanel::resetSimulationEvent, this, &MainWindow::onResetSimulationEvent);
	connect(m_controlPanel, &ControlPanel::changeScenarioEvent, this, &MainWindow::onChangeScenarioEvent);
	connect(m_controlPanel, &ControlPanel::showParticlesEvent, this, &MainWindow::onShowParticlesEvent);
	connect(m_controlPanel, &ControlPanel::showMeshEvent, this, &MainWindow::onShowMeshEvent);
	connect(m_controlPanel, &ControlPanel::changeTimestepEvent, this, &MainWindow::onChangeTimestepEvent);
	connect(m_controlPanel, &ControlPanel::changeSolverIterationsEvent, this, &MainWindow::onChangeSolverIterationsEvent);
	connect(m_controlPanel, &ControlPanel::changeSubstepsEvent, this, &MainWindow::onChangeSubstepsEvent);
	connect(m_controlPanel, &ControlPanel::changeGravityEvent, this, &MainWindow::onChangeGravityEvent);
	connect(m_controlPanel, &ControlPanel::changeDampingEvent, this, &MainWindow::onChangeDampingEvent);
	connect(m_controlPanel, &ControlPanel::changeParticleSleepingThresholdEvent, this, &MainWindow::onChangeParticleSleepingThresholdEvent);
	connect(m_controlPanel, &ControlPanel::changeStaticFrictionEvent, this, &MainWindow::onChangeStaticFrictionEvent);
	connect(m_controlPanel, &ControlPanel::changeDynamicFrictionEvent, this, &MainWindow::onChangeDynamicFrictionEvent);
	connect(m_controlPanel, &ControlPanel::changeDeformationThreshold, this, &MainWindow::onChangeDeformationThresholdEvent);
	connect(m_controlPanel, &ControlPanel::changeRotExtractionIterationsEvent, this, &MainWindow::onChangeRotExtractionIterationsEvent);
	connect(m_controlPanel, &ControlPanel::changeStiffnessSoftLinearEvent, this, &MainWindow::onChangeStiffnessSoftLinearEvent);
	connect(m_controlPanel, &ControlPanel::changeBetaSoftLinearEvent, this, &MainWindow::onChangeBetaSoftLinearEvent);
	connect(m_controlPanel, &ControlPanel::enableVolumeConservationEvent, this, &MainWindow::onEnableVolumeConservationEvent);
	connect(m_controlPanel, &ControlPanel::changeStiffnessSoftQuadraticEvent, this, &MainWindow::onChangeStiffnessSoftQuadraticEvent);
	connect(m_controlPanel, &ControlPanel::changeBetaSoftQuadraticEvent, this, &MainWindow::onChangeBetaSoftQuadraticEvent);

	m_controlPanel->initialize();

	std::shared_ptr<Camera> camera = m_ui.openGLWidget->getCamera();
	camera->setPosition(QVector3D(0.f, 3.f, 30.f));

	m_cameraController = std::make_shared<CameraController>(camera);
	m_cameraController->setMovementSpeed(10.f);
	m_cameraController->setMovementBoost(3.f);
	m_cameraController->setRotationSpeed(0.1f);

	m_particleDragger = std::make_shared<ParticleDragger>(m_pbdSolver ,camera);

	QTimer* showElapsedTimeTimer = new QTimer();
	connect(showElapsedTimeTimer, &QTimer::timeout, this, &MainWindow::showElapsedTime);
	showElapsedTimeTimer->start(100);

	QTimer* updateTimer = new QTimer();
	connect(updateTimer, &QTimer::timeout, this, &MainWindow::update);
	updateTimer->start(0);

	m_elapsedTimer.start();
}

void MainWindow::update()
{
	float deltaTime = (float)m_elapsedTimer.nsecsElapsed() / 1000000000.f;
	m_elapsedTimer.restart();

	updateFps(deltaTime);
	m_cameraController->update(deltaTime);

	if (m_hasSimulationStarted)
	{
		if (m_wasScenarioReset)
		{
			m_elapsedTimestep = 0.f;
			m_wasScenarioReset = false;
		}
		else
		{
			m_elapsedTimestep += deltaTime;
		}

		if (m_elapsedTimestep > m_targetTimestep)
		{
			m_currentScenario->update(m_targetTimestep);
			m_pbdSolver->update(m_targetTimestep);
			m_elapsedSimulationTime += m_targetTimestep;
			m_elapsedTimestep -= m_targetTimestep;
		}
	}

	m_ui.openGLWidget->update();
}

void MainWindow::onStartSimulationEvent()
{
	m_hasSimulationStarted = true;
}

void MainWindow::onPauseSimulationEvent()
{
	m_hasSimulationStarted = false;
	m_elapsedTimestep = 0.f;
}

void MainWindow::onStepSimulationEvent()
{
	m_currentScenario->update(m_targetTimestep);
	m_pbdSolver->update(m_targetTimestep);
	m_elapsedSimulationTime += m_targetTimestep;
	m_controlPanel->onElapsedTimeChanged(m_elapsedSimulationTime);
}

void MainWindow::onResetSimulationEvent()
{
	if (!m_currentScenario)
		return;

	m_currentScenario->initialize();
	m_controlPanel->onElapsedTimeChanged(m_elapsedSimulationTime);
	m_controlPanel->onParticleCountChanged(m_pbdSolver->getParticleCount());
	m_controlPanel->onShapeCountChanged(m_pbdSolver->getShapeCount());
	m_controlPanel->onClusterCountChanged(m_pbdSolver->getClusterCount());
	m_controlPanel->onVertexCountChanged(m_pbdSolver->getVertexCount());
	m_elapsedSimulationTime = 0.f;
	m_wasScenarioReset = true;
}

void MainWindow::onChangeScenarioEvent(std::shared_ptr<Scenario> changedScenario)
{
	m_currentScenario = changedScenario;
	m_currentScenario->setPBDSolver(m_pbdSolver);
	onResetSimulationEvent();
}

void MainWindow::onShowParticlesEvent(bool isEnabled)
{
	m_ui.openGLWidget->enableParticleRendering(isEnabled);
}

void MainWindow::onShowMeshEvent(bool isEnabled)
{
	m_ui.openGLWidget->enableShapeMeshRendering(isEnabled);
}

void MainWindow::onChangeTimestepEvent(float timestep)
{
	m_targetTimestep = timestep;
}

void MainWindow::onChangeSolverIterationsEvent(int solverIterations)
{
	m_pbdSolver->setSolverIterations(solverIterations);
}

void MainWindow::onChangeSubstepsEvent(int substeps)
{
	m_pbdSolver->setSubsteps(substeps);
}

void MainWindow::onChangeGravityEvent(QVector3D gravity)
{
	m_pbdSolver->setGravity(gravity);
}

void MainWindow::onChangeDampingEvent(float damping)
{
	m_pbdSolver->setDamping(damping);
}

void MainWindow::onChangeParticleSleepingThresholdEvent(float particleSleepingThreshold)
{
	m_pbdSolver->setParticleSleepingThreshold(particleSleepingThreshold);
}

void MainWindow::onChangeStaticFrictionEvent(float staticFriction)
{
	m_pbdSolver->setStaticFriction(staticFriction);
}

void MainWindow::onChangeDynamicFrictionEvent(float dynamicFriction)
{
	m_pbdSolver->setDynamicFriction(dynamicFriction);
}

void MainWindow::onChangeDeformationThresholdEvent(float deformationThreshold)
{
	m_pbdSolver->setDeformationThreshold(deformationThreshold);
}

void MainWindow::onChangeRotExtractionIterationsEvent(int rotExtractionIterations)
{
	m_pbdSolver->setRotExtractionIterations(rotExtractionIterations);
}

void MainWindow::onChangeStiffnessSoftLinearEvent(float stiffness)
{
	m_pbdSolver->setStiffnessSoftLinear(stiffness);
}

void MainWindow::onChangeBetaSoftLinearEvent(float beta)
{
	m_pbdSolver->setBetaSoftLinear(beta);
}

void MainWindow::onEnableVolumeConservationEvent(bool isEnabled)
{
	m_pbdSolver->enableVolumeConservationSoftLinear(isEnabled);
}

void MainWindow::onChangeStiffnessSoftQuadraticEvent(float stiffness)
{
	m_pbdSolver->setStiffnessSoftQuadratic(stiffness);
}

void MainWindow::onChangeBetaSoftQuadraticEvent(float beta)
{
	m_pbdSolver->setBetaSoftQuadratic(beta);
}

void MainWindow::showElapsedTime()
{
	m_controlPanel->onElapsedTimeChanged(m_elapsedSimulationTime);
}

void MainWindow::resizeEvent(QResizeEvent * resizeEvent)
{
	m_controlPanel->resize(resizeEvent->size());
}

void MainWindow::mouseMoveEvent(QMouseEvent * event)
{
	m_cameraController->onMouseMoveEvent(event);
	m_particleDragger->onMouseMoveEvent(event);
}

void MainWindow::mousePressEvent(QMouseEvent * event)
{
	m_cameraController->onMousePressEvent(event);
	m_particleDragger->onMousePressEvent(event);
}

void MainWindow::mouseReleaseEvent(QMouseEvent * event)
{
	m_cameraController->onMouseReleaseEvent(event);
	m_particleDragger->onMouseReleaseEvent(event);
}

void MainWindow::keyPressEvent(QKeyEvent * event)
{
	m_cameraController->onKeyPressEvent(event);
}

void MainWindow::keyReleaseEvent(QKeyEvent * event)
{
	m_cameraController->onKeyReleaseEvent(event);
}

void MainWindow::updateFps(float deltaTime)
{
	if (m_fpsAccumulatedTime < m_fpsAveragingTime)
	{
		m_fpsAccumulatedTime += deltaTime;
		m_fpsFrameCount++;
	}
	else
	{
		m_fps = m_fpsFrameCount / m_fpsAccumulatedTime;
		setWindowTitle("FPS: " + QString::number(m_fps, 'f', 1));
		m_fpsFrameCount = 0;
		m_fpsAccumulatedTime = 0.f;
	}
}

