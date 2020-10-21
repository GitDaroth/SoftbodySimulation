#include "Scenario/ObstacleCourseScenario.h"

#include <Constraint/BoxConstraint.h>
#include <Constraint/RigidShapeConstraint.h>
#include <Constraint/SoftShapeLinearConstraint.h>
#include <Constraint/SoftShapeQuadraticConstraint.h>

const QString ObstacleCourseScenario::NAME = "ObstacleCourse";

ObstacleCourseScenario::ObstacleCourseScenario()
{
}

ObstacleCourseScenario::~ObstacleCourseScenario()
{
}

void ObstacleCourseScenario::onInitialize()
{
	m_dynamicObstacles.clear();
	m_elapsedTime = 0.f;
	
	std::shared_ptr<ShapeConstraint> shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/bunny_normal.pts", "assets/meshes/bunny_normal.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(-8.f, 15.f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/dragon.pts", "assets/meshes/dragon.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(-8.f, 15.f, 2.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/armadillo.pts", "assets/meshes/armadillo.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(-8.f, 17.f, -2.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/patrick.pts", "assets/meshes/patrick.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(-8.f, 16.5f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/octopus2.pts", "assets/meshes/octopus2.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(-8.f, 16.5f, 2.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/bunny_big.pts", "assets/meshes/bunny_big.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(-8.f, 15.f, -2.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	std::shared_ptr<BoxConstraint> boxBoundaryConstraint = std::make_shared<BoxConstraint>(QVector3D(-0.5f, 2.5f, 0.f), QVector3D(9.75f, 17.5f, 15.25f));
	boxBoundaryConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxBoundaryConstraint->setIsBoundary(true);
	m_pbdSolver->addConstraint(boxBoundaryConstraint);

	// moving wall 1 -----------------------------------------
	std::shared_ptr<BoxConstraint> boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(-8.f, 14.f, 0.f), QVector3D(2.f, 0.25f, 4.f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(-9.75f, 17.125f, 0.f), QVector3D(0.25f, 2.87f, 4.f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	m_dynamicObstacles.push_back(boxObstacleConstraint);
	// -------------------------------------------------------

	// moving wall 2 -----------------------------------------
	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(2.f, -3.5f, -11.25f), QVector3D(7.25f, 0.25f, 4.f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(2.f, -1.22f, -19.25f), QVector3D(7.25f, 2.f, 4.f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	m_dynamicObstacles.push_back(boxObstacleConstraint);
	// -------------------------------------------------------

	// moving wall 3 -----------------------------------------
	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(2.f, -3.5f, 11.25f), QVector3D(7.25f, 0.25f, 4.f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(2.f, -1.22f, 19.25f), QVector3D(7.25f, 2.f, 4.f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	m_dynamicObstacles.push_back(boxObstacleConstraint);
	// -------------------------------------------------------

	// ramp 1 -----------------------------------------
	QVector3D rampPosition = QVector3D(-2.661f, 12.033f, 0.f);
	boxObstacleConstraint = std::make_shared<BoxConstraint>(rampPosition, QVector3D(4.f, 0.25f, 4.f));
	boxObstacleConstraint->setRotation(-30.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(rampPosition + QVector3D(0.381f, 0.617f, 3.8f), QVector3D(4.f, 1.f, 0.25f));
	boxObstacleConstraint->setRotation(-30.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(rampPosition + QVector3D(0.381f, 0.617f, -3.8f), QVector3D(4.f, 1.f, 0.25f));
	boxObstacleConstraint->setRotation(-30.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	// ------------------------------------------------

	// ramp 2 -----------------------------------------
	rampPosition = QVector3D(5.f, 3.f, -3.4f);
	boxObstacleConstraint = std::make_shared<BoxConstraint>(rampPosition, QVector3D(4.f, 0.25f, 4.f));
	boxObstacleConstraint->setRotation(-30.f, QVector3D(1.f, 0.f, 0.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(rampPosition + QVector3D(3.8f, 0.617f, -0.381f), QVector3D(0.25, 1.f, 4.f));
	boxObstacleConstraint->setRotation(-30.f, QVector3D(1.f, 0.f, 0.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(rampPosition + QVector3D(-3.8f, 0.617f, -0.381f), QVector3D(0.25, 1.f, 4.f));
	boxObstacleConstraint->setRotation(-30.f, QVector3D(1.f, 0.f, 0.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	// ------------------------------------------------

	// ramp 3 -----------------------------------------
	rampPosition = QVector3D(5.f, 3.f, 3.4f);
	boxObstacleConstraint = std::make_shared<BoxConstraint>(rampPosition, QVector3D(4.f, 0.25f, 4.f));
	boxObstacleConstraint->setRotation(30.f, QVector3D(1.f, 0.f, 0.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(rampPosition + QVector3D(3.8f, 0.617f, 0.381f), QVector3D(0.25, 1.f, 4.f));
	boxObstacleConstraint->setRotation(30.f, QVector3D(1.f, 0.f, 0.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(rampPosition + QVector3D(-3.8f, 0.617f, 0.381f), QVector3D(0.25, 1.f, 4.f));
	boxObstacleConstraint->setRotation(30.f, QVector3D(1.f, 0.f, 0.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	// ------------------------------------------------

	// funnel -----------------------------------------
	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(6.5f, -5.f, 0.f), QVector3D(3.f, 0.25f, 7.f));
	boxObstacleConstraint->setRotation(30.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(-2.5f, -5.f, 0.f), QVector3D(3.f, 0.25f, 7.f));
	boxObstacleConstraint->setRotation(-30.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(1.95f, -5.f, -4.55f), QVector3D(7.f, 0.25f, 3.f));
	boxObstacleConstraint->setRotation(30.f, QVector3D(1.f, 0.f, 0.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(1.95f, -5.f, 4.55f), QVector3D(7.f, 0.25f, 3.f));
	boxObstacleConstraint->setRotation(-30.f, QVector3D(1.f, 0.f, 0.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(9.f, -5.f, 0.f), QVector3D(0.25f, 1.75f, 7.25f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(-5.f, -5.f, 0.f), QVector3D(0.25f, 1.75f, 7.25f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(2.f, -5.f, 7.f), QVector3D(7.25f, 1.75f, 0.25f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(2.f, -5.f, -7.f), QVector3D(7.25f, 1.75f, 0.25f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	// ------------------------------------------------

	// spinning box 1 -----------------------------------------
	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(2.5f, 10.f, 0.f), QVector3D(1.5f, 0.25f, 4.f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	m_dynamicObstacles.push_back(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(2.5f, 10.f, 0.f), QVector3D(1.5f, 0.25f, 4.f));
	boxObstacleConstraint->setRotation(90.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	m_dynamicObstacles.push_back(boxObstacleConstraint);
	// --------------------------------------------------------

	// spinning box 2 -----------------------------------------
	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(5.f, 8.f, 0.f), QVector3D(1.5f, 0.25f, 4.f));
	boxObstacleConstraint->setRotation(67.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	m_dynamicObstacles.push_back(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(5.f, 8.f, 0.f), QVector3D(1.5f, 0.25f, 4.f));
	boxObstacleConstraint->setRotation(90.f + 67.f, QVector3D(0.f, 0.f, 1.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	m_dynamicObstacles.push_back(boxObstacleConstraint);
	// --------------------------------------------------------

	// spinning box 3 -----------------------------------------
	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(2.f, -8.5f, 1.7f), QVector3D(3.f, 0.25f, 1.5f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(1.f, 0.f, 0.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	m_dynamicObstacles.push_back(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(2.f, -8.5f, 1.7f), QVector3D(3.f, 0.25f, 1.5f));
	boxObstacleConstraint->setRotation(90.f, QVector3D(1.f, 0.f, 0.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	m_dynamicObstacles.push_back(boxObstacleConstraint);
	// --------------------------------------------------------

	// spinning box 4 -----------------------------------------
	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(2.f, -8.5f, -1.7f), QVector3D(3.f, 0.25f, 1.5f));
	boxObstacleConstraint->setRotation(0.f, QVector3D(1.f, 0.f, 0.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	m_dynamicObstacles.push_back(boxObstacleConstraint);

	boxObstacleConstraint = std::make_shared<BoxConstraint>(QVector3D(2.f, -8.5f, -1.7f), QVector3D(3.f, 0.25f, 1.5f));
	boxObstacleConstraint->setRotation(90.f, QVector3D(1.f, 0.f, 0.f));
	boxObstacleConstraint->setIsBoundary(false);
	m_pbdSolver->addConstraint(boxObstacleConstraint);
	m_dynamicObstacles.push_back(boxObstacleConstraint);
	// --------------------------------------------------------
}

void ObstacleCourseScenario::onUpdate(float deltaTime)
{
	m_elapsedTime += deltaTime;
	m_dynamicObstacles[0]->setPosition(3.5f * 0.5f * (-cosf(m_elapsedTime) + 1.f) * QVector3D(1.f, 0.f, 0.f) + QVector3D(-9.75f, 17.125f, 0.f));
	m_dynamicObstacles[1]->setPosition(8.f * 0.5f * (-cosf(m_elapsedTime) + 1.f) * QVector3D(0.f, 0.f, 1.f) + QVector3D(2.f, -1.22f, -19.25f));
	m_dynamicObstacles[2]->setPosition(-8.f * 0.5f * (-cosf(m_elapsedTime) + 1.f) * QVector3D(0.f, 0.f, 1.f) + QVector3D(2.f, -1.22f, 19.25f));

	m_dynamicObstacles[3]->setRotation(-100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_dynamicObstacles[4]->setRotation(90.f - 100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_dynamicObstacles[5]->setRotation(67.f + 100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_dynamicObstacles[6]->setRotation(90.f + 67.f + 100.f * m_elapsedTime, QVector3D(0.f, 0.f, 1.f));
	m_dynamicObstacles[7]->setRotation(-100.f * m_elapsedTime, QVector3D(1.f, 0.f, 0.f));
	m_dynamicObstacles[8]->setRotation(90.f - 100.f * m_elapsedTime, QVector3D(1.f, 0.f, 0.f));
	m_dynamicObstacles[9]->setRotation(100.f * m_elapsedTime, QVector3D(1.f, 0.f, 0.f));
	m_dynamicObstacles[10]->setRotation(90.f + 100.f * m_elapsedTime, QVector3D(1.f, 0.f, 0.f));
}