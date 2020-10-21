#include "Widget/ControlPanel.h"

#include <QGraphicsOpacityEffect>
#include "Scenario/ObstacleCourseScenario.h"
#include "Scenario/ShapeTypeComparisonScenario.h"
#include "Scenario/StairsScenario.h"
#include "Scenario/WashingMachineScenario.h"
#include "Scenario/SpinningBoxesScenario.h"
#include "Scenario/MovingWallScenario.h"
#include "Scenario/BlockScenario.h"

ControlPanel::ControlPanel(QWidget *parent)
	: QWidget(parent)
{
	m_ui.setupUi(this);
}

ControlPanel::~ControlPanel()
{
}

void ControlPanel::initialize()
{
	QPalette palette;
	palette.setColor(QPalette::Background, QColor(0, 0, 0, 0));
	m_ui.scrollArea->setPalette(palette);

	palette.setColor(QPalette::Background, QColor(50, 50, 50, 130));
	m_ui.frame->setPalette(palette);
	m_ui.simulationInfoFrame->setPalette(palette);

	m_ui.stepButton->setDisabled(false);

	connect(m_ui.startPauseButton, &QPushButton::clicked, this, &ControlPanel::onStartPauseButtonClicked);
	connect(m_ui.stepButton, &QPushButton::clicked, this, &ControlPanel::onStepButtonClicked);
	connect(m_ui.resetButton, &QPushButton::clicked, this, &ControlPanel::onResetButtonClicked);
	connect(m_ui.showParticlesCheckBox, &QAbstractButton::clicked, this, &ControlPanel::onShowParticlesCheckBoxClicked);
	connect(m_ui.showMeshCheckBox, &QAbstractButton::clicked, this, &ControlPanel::onShowMeshCheckBoxClicked);
	connect(m_ui.timestepSlider, &QSlider::valueChanged, this, &ControlPanel::onTimestepSliderChanged);
	connect(m_ui.solverIterationsSlider, &QSlider::valueChanged, this, &ControlPanel::onSolverIterationsSliderChanged);
	connect(m_ui.substepsSlider, &QSlider::valueChanged, this, &ControlPanel::onSubstepsSliderChanged);
	connect(m_ui.gravityXSlider, &QSlider::valueChanged, this, &ControlPanel::onGravityXSliderChanged);
	connect(m_ui.gravityYSlider, &QSlider::valueChanged, this, &ControlPanel::onGravityYSliderChanged);
	connect(m_ui.gravityZSlider, &QSlider::valueChanged, this, &ControlPanel::onGravityZSliderChanged);
	connect(m_ui.dampingSlider, &QSlider::valueChanged, this, &ControlPanel::onDampingSliderChanged);
	connect(m_ui.particleSleepingThresholdSlider, &QSlider::valueChanged, this, &ControlPanel::onParticleSleepingThresholdSliderChanged);
	connect(m_ui.staticFrictionSlider, &QSlider::valueChanged, this, &ControlPanel::onStaticFrictionSliderChanged);
	connect(m_ui.dynamicFrictionSlider, &QSlider::valueChanged, this, &ControlPanel::onDynamicFrictionSliderChanged);
	connect(m_ui.deformationThresholdSlider, &QSlider::valueChanged, this, &ControlPanel::onDeformationThresholdSliderChanged);
	connect(m_ui.rotExtractionIterationsSlider, &QSlider::valueChanged, this, &ControlPanel::onRotExtractionIterationsSliderChanged);
	connect(m_ui.stiffnessSoftLinearSlider, &QSlider::valueChanged, this, &ControlPanel::onStiffnessSoftLinearSliderChanged);
	connect(m_ui.betaSoftLinearSlider, &QSlider::valueChanged, this, &ControlPanel::onBetaSoftLinearSliderChanged);
	connect(m_ui.enableVolumeConservationCheckBox, &QAbstractButton::clicked, this, &ControlPanel::onEnableVolumeConservationCheckBoxClicked);
	connect(m_ui.stiffnessSoftQuadraticSlider, &QSlider::valueChanged, this, &ControlPanel::onStiffnessSoftQuadraticSliderChanged);
	connect(m_ui.betaSoftQuadraticSlider, &QSlider::valueChanged, this, &ControlPanel::onBetaSoftQuadraticSliderChanged);

	m_ui.scenarioSelectionBox->addItem(ObstacleCourseScenario::NAME);
	m_ui.scenarioSelectionBox->addItem(ShapeTypeComparisonScenario::NAME);
	m_ui.scenarioSelectionBox->addItem(BlockScenario::NAME);
	m_ui.scenarioSelectionBox->addItem(MovingWallScenario::NAME);
	m_ui.scenarioSelectionBox->addItem(SpinningBoxesScenario::NAME);
	m_ui.scenarioSelectionBox->addItem(WashingMachineScenario::NAME);
	m_ui.scenarioSelectionBox->addItem(StairsScenario::NAME);

	//there are 2 currentIndexChanged signals for QComboBox (with parameter "int" and "QString") -> static_cast is needed 
	connect(m_ui.scenarioSelectionBox, static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged), this, &ControlPanel::onScenarioSelectionChanged);

	QString firstScenarioText = m_ui.scenarioSelectionBox->currentText();
	onScenarioSelectionChanged(firstScenarioText);
}

void ControlPanel::onStartPauseButtonClicked(bool checked)
{
	if (checked)
	{
		m_ui.startPauseButton->setText("Pause");
		m_ui.stepButton->setDisabled(true);
		emit startSimulationEvent();
	}
	else
	{
		m_ui.startPauseButton->setText("Start");
		m_ui.stepButton->setDisabled(false);
		emit pauseSimulationEvent();
	}
}

void ControlPanel::onStepButtonClicked()
{
	emit stepSimulationEvent();
}

void ControlPanel::onResetButtonClicked()
{
	emit resetSimulationEvent();
}

void ControlPanel::onScenarioSelectionChanged(const QString& text)
{
	std::shared_ptr<Scenario> changedScenario;

	if (text == ObstacleCourseScenario::NAME)
		changedScenario = std::make_shared<ObstacleCourseScenario>();
	else if (text == ShapeTypeComparisonScenario::NAME)
		changedScenario = std::make_shared<ShapeTypeComparisonScenario>();
	else if (text == StairsScenario::NAME)
		changedScenario = std::make_shared<StairsScenario>();
	else if (text == WashingMachineScenario::NAME)
		changedScenario = std::make_shared<WashingMachineScenario>();
	else if (text == SpinningBoxesScenario::NAME)
		changedScenario = std::make_shared<SpinningBoxesScenario>();
	else if (text == MovingWallScenario::NAME)
		changedScenario = std::make_shared<MovingWallScenario>();
	else if (text == BlockScenario::NAME)
		changedScenario = std::make_shared<BlockScenario>();

	emit changeScenarioEvent(changedScenario);
}

void ControlPanel::onShowParticlesCheckBoxClicked(bool checked)
{
	emit showParticlesEvent(checked);
}

void ControlPanel::onShowMeshCheckBoxClicked(bool checked)
{
	emit showMeshEvent(checked);
}

void ControlPanel::onTimestepSliderChanged(int value)
{
	m_ui.timestepValueLabel->setText(QString::number(value * m_timestepSliderFactor * 1000.f, 'f', 1) + " ms");
	emit changeTimestepEvent(value * m_timestepSliderFactor);
}

void ControlPanel::onSolverIterationsSliderChanged(int value)
{
	m_ui.solverIterationsValueLabel->setText(QString::number(value));
	emit changeSolverIterationsEvent(value);
}

void ControlPanel::onSubstepsSliderChanged(int value)
{
	m_ui.substepsValueLabel->setText(QString::number(value));
	emit changeSubstepsEvent(value);
}

void ControlPanel::onGravityXSliderChanged(int value)
{
	m_ui.gravityXValueLabel->setText(QString::number(value * m_gravitySliderFactor, 'f', 1) + QString::fromLatin1(" m/s²"));

	QVector3D gravity;
	gravity.setX(m_ui.gravityXSlider->value() * m_gravitySliderFactor);
	gravity.setY(m_ui.gravityYSlider->value() * m_gravitySliderFactor);
	gravity.setZ(m_ui.gravityZSlider->value() * m_gravitySliderFactor);
	emit changeGravityEvent(gravity);
}

void ControlPanel::onGravityYSliderChanged(int value)
{
	m_ui.gravityYValueLabel->setText(QString::number(value * m_gravitySliderFactor, 'f', 1) + QString::fromLatin1(" m/s²"));

	QVector3D gravity;
	gravity.setX(m_ui.gravityXSlider->value() * m_gravitySliderFactor);
	gravity.setY(m_ui.gravityYSlider->value() * m_gravitySliderFactor);
	gravity.setZ(m_ui.gravityZSlider->value() * m_gravitySliderFactor);
	emit changeGravityEvent(gravity);
}

void ControlPanel::onGravityZSliderChanged(int value)
{
	m_ui.gravityZValueLabel->setText(QString::number(value * m_gravitySliderFactor, 'f', 1) + QString::fromLatin1(" m/s²"));

	QVector3D gravity;
	gravity.setX(m_ui.gravityXSlider->value() * m_gravitySliderFactor);
	gravity.setY(m_ui.gravityYSlider->value() * m_gravitySliderFactor);
	gravity.setZ(m_ui.gravityZSlider->value() * m_gravitySliderFactor);
	emit changeGravityEvent(gravity);
}

void ControlPanel::onDampingSliderChanged(int value)
{
	m_ui.dampingValueLabel->setText(QString::number(value * m_dampingSliderFactor, 'f', 2));
	emit changeDampingEvent(value * m_dampingSliderFactor);
}

void ControlPanel::onParticleSleepingThresholdSliderChanged(int value)
{
	m_ui.particleSleepingThresholdValueLabel->setText(QString::number(value * m_particleSleepingThresholdSliderFactor, 'f', 4) + " m/s");
	emit changeParticleSleepingThresholdEvent(value * m_particleSleepingThresholdSliderFactor);
}

void ControlPanel::onStaticFrictionSliderChanged(int value)
{
	m_ui.staticFrictionValueLabel->setText(QString::number(value * m_staticFrictionSliderFactor, 'f', 2));
	emit changeStaticFrictionEvent(value * m_staticFrictionSliderFactor);
}

void ControlPanel::onDynamicFrictionSliderChanged(int value)
{
	m_ui.dynamicFrictionValueLabel->setText(QString::number(value * m_dynamicFrictionSliderFactor, 'f', 2));
	emit changeDynamicFrictionEvent(value * m_dynamicFrictionSliderFactor);
}

void ControlPanel::onDeformationThresholdSliderChanged(int value)
{
	m_ui.deformationThresholdValueLabel->setText(QString::number(value));
	emit changeDeformationThreshold(value);
}

void ControlPanel::onRotExtractionIterationsSliderChanged(int value)
{
	m_ui.rotExtractionIterationsValueLabel->setText(QString::number(value));
	emit changeRotExtractionIterationsEvent(value);
}

void ControlPanel::onStiffnessSoftLinearSliderChanged(int value)
{
	m_ui.stiffnessSoftLinearValueLabel->setText(QString::number(value * m_stiffnessSliderFactor, 'f', 1));
	emit changeStiffnessSoftLinearEvent(value * m_stiffnessSliderFactor);
}

void ControlPanel::onBetaSoftLinearSliderChanged(int value)
{
	m_ui.betaSoftLinearValueLabel->setText(QString::number(value * m_betaSliderFactor, 'f', 1));
	emit changeBetaSoftLinearEvent(value * m_betaSliderFactor);
}

void ControlPanel::onEnableVolumeConservationCheckBoxClicked(bool checked)
{
	emit enableVolumeConservationEvent(checked);
}

void ControlPanel::onStiffnessSoftQuadraticSliderChanged(int value)
{
	m_ui.stiffnessSoftQuadraticValueLabel->setText(QString::number(value * m_stiffnessSliderFactor, 'f', 1));
	emit changeStiffnessSoftQuadraticEvent(value * m_stiffnessSliderFactor);
}

void ControlPanel::onBetaSoftQuadraticSliderChanged(int value)
{
	m_ui.betaSoftQuadraticValueLabel->setText(QString::number(value * m_betaSliderFactor, 'f', 1));
	emit changeBetaSoftQuadraticEvent(value * m_betaSliderFactor);
}

void ControlPanel::onElapsedTimeChanged(float elapsedTime)
{
	m_ui.elapsedTimeLabel->setText("Elapsed Simulation Time: " + QString::number(elapsedTime, 'f', 4) + " s");
}

void ControlPanel::onParticleCountChanged(int particleCount)
{
	m_ui.particleCountLabel->setText("#Particles: " + QString::number(particleCount));
}

void ControlPanel::onShapeCountChanged(int shapeCount)
{
	m_ui.shapeCountLabel->setText("#Shapes: " + QString::number(shapeCount));
}

void ControlPanel::onClusterCountChanged(int clusterCount)
{
	m_ui.clusterCountLabel->setText("#Clusters: " + QString::number(clusterCount));
}

void ControlPanel::onVertexCountChanged(int vertexCount)
{
	m_ui.vertexCountLabel->setText("#Vertices: " + QString::number(vertexCount));
}
