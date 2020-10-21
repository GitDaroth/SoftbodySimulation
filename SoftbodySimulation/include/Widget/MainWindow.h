#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_MainWindow.h"
#include "ControlPanel.h"
#include <QElapsedTimer>
#include "Camera/CameraController.h"
#include <PBDSolver.h>
#include "ParticleDragger.h"
#include "Scenario/Scenario.h"

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = Q_NULLPTR);

public slots:
	void update();

	void onStartSimulationEvent();
	void onPauseSimulationEvent();
	void onStepSimulationEvent();
	void onResetSimulationEvent();
	void onChangeScenarioEvent(std::shared_ptr<Scenario> changedScenario);
	void onShowParticlesEvent(bool isEnabled);
	void onShowMeshEvent(bool isEnabled);
	void onChangeTimestepEvent(float timestep);
	void onChangeSolverIterationsEvent(int solverIterations);
	void onChangeSubstepsEvent(int substeps);
	void onChangeGravityEvent(QVector3D gravity);
	void onChangeDampingEvent(float damping);
	void onChangeParticleSleepingThresholdEvent(float particleSleepingThreshold);
	void onChangeStaticFrictionEvent(float staticFriction);
	void onChangeDynamicFrictionEvent(float dynamicFriction);
	void onChangeDeformationThresholdEvent(float deformationThreshold);
	void onChangeRotExtractionIterationsEvent(int rotExtractionIterations);
	void onChangeStiffnessSoftLinearEvent(float stiffness);
	void onChangeBetaSoftLinearEvent(float beta);
	void onEnableVolumeConservationEvent(bool isEnabled);
	void onChangeStiffnessSoftQuadraticEvent(float stiffness);
	void onChangeBetaSoftQuadraticEvent(float beta);

	void showElapsedTime();

protected:
	virtual void resizeEvent(QResizeEvent* resizeEvent) override;
	virtual void mouseMoveEvent(QMouseEvent* event) override;
	virtual void mousePressEvent(QMouseEvent* event) override;
	virtual void mouseReleaseEvent(QMouseEvent* event) override;
	virtual void keyPressEvent(QKeyEvent* event) override;
	virtual void keyReleaseEvent(QKeyEvent* event) override;

private:
	void updateFps(float deltaTime);

	Ui::MainWindowClass m_ui;
	ControlPanel* m_controlPanel;
	QElapsedTimer m_elapsedTimer;
	std::shared_ptr<CameraController> m_cameraController;
	std::shared_ptr<PBDSolver> m_pbdSolver;
	std::shared_ptr<ParticleDragger> m_particleDragger;
	std::shared_ptr<Scenario> m_currentScenario;

	float m_fps;
	float m_fpsAveragingTime;
	float m_fpsAccumulatedTime;
	unsigned int m_fpsFrameCount;

	float m_elapsedSimulationTime;
	float m_targetTimestep;
	float m_elapsedTimestep;
	
	bool m_hasSimulationStarted;
	bool m_wasScenarioReset;
};
