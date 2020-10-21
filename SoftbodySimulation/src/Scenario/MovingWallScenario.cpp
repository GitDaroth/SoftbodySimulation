#include "Scenario/MovingWallScenario.h"

#include <Constraint/BoxConstraint.h>
#include <Constraint/RigidShapeConstraint.h>
#include <Constraint/SoftShapeLinearConstraint.h>
#include <Constraint/SoftShapeQuadraticConstraint.h>

const QString MovingWallScenario::NAME = "MovingWall";

MovingWallScenario::MovingWallScenario()
{
}

MovingWallScenario::~MovingWallScenario()
{
}

void MovingWallScenario::onInitialize()
{
	m_elapsedTime = 0.f;

	std::shared_ptr<ShapeConstraint> shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/bunny_normal.pts", "assets/meshes/bunny_normal.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(2.f, 4.f, -3.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
	{
		particle->setVelocity(QVector3D(1.5f, 0.f, 0.f));
		m_pbdSolver->addParticle(particle);
	}

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/octopus2.pts", "assets/meshes/octopus2.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(2.f, 4.f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
	{
		particle->setVelocity(QVector3D(1.5f, 0.f, 0.f));
		m_pbdSolver->addParticle(particle);
	}

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/octopus2.pts", "assets/meshes/octopus2.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(2.f, 4.f, 3.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
	{
		particle->setVelocity(QVector3D(1.5f, 0.f, 0.f));
		m_pbdSolver->addParticle(particle);
	}

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/dragon.pts", "assets/meshes/dragon.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(2.f, 6.f, -3.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
	{
		particle->setVelocity(QVector3D(1.5f, 0.f, 0.f));
		m_pbdSolver->addParticle(particle);
	}

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/armadillo.pts", "assets/meshes/armadillo.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(2.f, 6.f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
	{
		particle->setVelocity(QVector3D(1.5f, 0.f, 0.f));
		m_pbdSolver->addParticle(particle);
	}

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/armadillo.pts", "assets/meshes/armadillo.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(2.f, 6.f, 3.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
	{
		particle->setVelocity(QVector3D(1.5f, 0.f, 0.f));
		m_pbdSolver->addParticle(particle);
	}

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/patrick.pts", "assets/meshes/patrick.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(0.f, 6.f, -3.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
	{
		particle->setVelocity(QVector3D(1.5f, 0.f, 0.f));
		m_pbdSolver->addParticle(particle);
	}

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/dragon.pts", "assets/meshes/dragon.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(0.f, 6.f, 0.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
	{
		particle->setVelocity(QVector3D(1.5f, 0.f, 0.f));
		m_pbdSolver->addParticle(particle);
	}

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/bunny_small.pts", "assets/meshes/bunny_small.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(0.f, 6.f, 3.f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
	{
		particle->setVelocity(QVector3D(1.5f, 0.f, 0.f));
		m_pbdSolver->addParticle(particle);
	}

	std::shared_ptr<BoxConstraint> boxBoundaryConstraint = std::make_shared<BoxConstraint>(QVector3D(-1.f, 5.f, 0.f), QVector3D(4.f, 4.f, 4.f));
	boxBoundaryConstraint->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	boxBoundaryConstraint->setIsBoundary(true);
	m_pbdSolver->addConstraint(boxBoundaryConstraint);

	m_movingWall = std::make_shared<BoxConstraint>(QVector3D(-4.f, 5.f, 0.f), QVector3D(1.f, 4.f, 4.f));
	m_movingWall->setRotation(0.f, QVector3D(0.f, 0.f, 1.f));
	m_movingWall->setIsBoundary(false);
	m_pbdSolver->addConstraint(m_movingWall);
}

void MovingWallScenario::onUpdate(float deltaTime)
{
	m_elapsedTime += deltaTime;
	m_movingWall->setPosition(5.f * 0.5f * (-cosf(m_elapsedTime) + 1.f) * QVector3D(1.f, 0.f, 0.f) + QVector3D(-4.f, 5.f, 0.f));
}