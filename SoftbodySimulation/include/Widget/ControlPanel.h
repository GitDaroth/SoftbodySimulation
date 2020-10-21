#pragma once

#include <QWidget>
#include "ui_ControlPanel.h"
#include "Scenario/Scenario.h"

class ControlPanel : public QWidget
{
	Q_OBJECT

public:
	ControlPanel(QWidget *parent = Q_NULLPTR);
	~ControlPanel();

	void initialize();

	void onElapsedTimeChanged(float elapsedTime);
	void onParticleCountChanged(int particleCount);
	void onShapeCountChanged(int shapeCount);
	void onClusterCountChanged(int clusterCount);
	void onVertexCountChanged(int vertexCount);

signals:
	void startSimulationEvent();
	void pauseSimulationEvent();
	void stepSimulationEvent();
	void resetSimulationEvent();
	void changeScenarioEvent(std::shared_ptr<Scenario> changedScenario);
	void showParticlesEvent(bool isEnabled);
	void showMeshEvent(bool isEnabled);
	void changeTimestepEvent(float timestep);
	void changeSolverIterationsEvent(int solverIterations);
	void changeSubstepsEvent(int substeps);
	void changeGravityEvent(QVector3D gravity);
	void changeDampingEvent(float damping);
	void changeParticleSleepingThresholdEvent(float particleSleepingThreshold);
	void changeStaticFrictionEvent(float staticFriction);
	void changeDynamicFrictionEvent(float dynamicFriction);
	void changeDeformationThreshold(float deformationThreshold);
	void changeRotExtractionIterationsEvent(int rotExtractionIterations);
	void changeStiffnessSoftLinearEvent(float stiffness);
	void changeBetaSoftLinearEvent(float beta);
	void enableVolumeConservationEvent(bool isEnabled);
	void changeStiffnessSoftQuadraticEvent(float stiffness);
	void changeBetaSoftQuadraticEvent(float beta);

private slots:
	void onStartPauseButtonClicked(bool checked);
	void onStepButtonClicked();
	void onResetButtonClicked();
	void onScenarioSelectionChanged(const QString& text);
	void onShowParticlesCheckBoxClicked(bool checked);
	void onShowMeshCheckBoxClicked(bool checked);
	void onTimestepSliderChanged(int value);
	void onSolverIterationsSliderChanged(int value);
	void onSubstepsSliderChanged(int value);
	void onGravityXSliderChanged(int value);
	void onGravityYSliderChanged(int value);
	void onGravityZSliderChanged(int value);
	void onDampingSliderChanged(int value);
	void onParticleSleepingThresholdSliderChanged(int value);
	void onStaticFrictionSliderChanged(int value);
	void onDynamicFrictionSliderChanged(int value);
	void onDeformationThresholdSliderChanged(int value);
	void onRotExtractionIterationsSliderChanged(int value);
	void onStiffnessSoftLinearSliderChanged(int value);
	void onBetaSoftLinearSliderChanged(int value);
	void onEnableVolumeConservationCheckBoxClicked(bool checked);
	void onStiffnessSoftQuadraticSliderChanged(int value);
	void onBetaSoftQuadraticSliderChanged(int value);

private:
	Ui::ControlPanel m_ui;

	// converts the int value of the respective sliders into the real float value
	float m_timestepSliderFactor = 0.0001f;
	float m_gravitySliderFactor = 0.1f;
	float m_dampingSliderFactor = 0.01f;
	float m_particleSleepingThresholdSliderFactor = 0.0001f;
	float m_staticFrictionSliderFactor = 0.01f;
	float m_dynamicFrictionSliderFactor = 0.01f;
	float m_stiffnessSliderFactor = 0.1f;
	float m_betaSliderFactor = 0.1f;
};
