#include "Scenario/ShapeTypeComparisonScenario.h"

#include <Constraint/BoxConstraint.h>
#include <Constraint/RigidShapeConstraint.h>
#include <Constraint/SoftShapeLinearConstraint.h>
#include <Constraint/SoftShapeQuadraticConstraint.h>

const QString ShapeTypeComparisonScenario::NAME = "Shape Type Comparison";

ShapeTypeComparisonScenario::ShapeTypeComparisonScenario()
{
}

ShapeTypeComparisonScenario::~ShapeTypeComparisonScenario()
{
}

void ShapeTypeComparisonScenario::onInitialize()
{
	std::shared_ptr<ShapeConstraint> shapeConstraint = std::make_shared<RigidShapeConstraint>("assets/points/bunny_normal.pts", "assets/meshes/bunny_normal.obj");
	shapeConstraint->initialize(false);
	shapeConstraint->setStartPosition(QVector3D(-3.f, 7.f, 1.5f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/bunny_normal.pts", "assets/meshes/bunny_normal.obj");
	shapeConstraint->initialize(false);
	shapeConstraint->setStartPosition(QVector3D(0.f, 7.f, 1.5f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeQuadraticConstraint>("assets/points/bunny_normal.pts", "assets/meshes/bunny_normal.obj");
	shapeConstraint->initialize(false);
	shapeConstraint->setStartPosition(QVector3D(3.f, 7.f, 1.5f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeLinearConstraint>("assets/points/bunny_normal.pts", "assets/meshes/bunny_normal.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(0.f, 7.f, -1.5f));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	shapeConstraint = std::make_shared<SoftShapeQuadraticConstraint>("assets/points/bunny_normal.pts", "assets/meshes/bunny_normal.obj");
	shapeConstraint->initialize(true);
	shapeConstraint->setStartPosition(QVector3D(3.f, 7.f, -1.5));
	m_pbdSolver->addConstraint(shapeConstraint);
	for (auto particle : shapeConstraint->getParticles())
		m_pbdSolver->addParticle(particle);

	std::shared_ptr<BoxConstraint> boxBoundaryConstraint = std::make_shared<BoxConstraint>(QVector3D(0.f, 5.f, 0.f), QVector3D(4.f, 5.f, 4.f));
	boxBoundaryConstraint->setRotation(0.f, QVector3D(1.f, 0.f, 0.f));
	boxBoundaryConstraint->setIsBoundary(true);
	m_pbdSolver->addConstraint(boxBoundaryConstraint);
}

void ShapeTypeComparisonScenario::onUpdate(float deltaTime)
{
}
