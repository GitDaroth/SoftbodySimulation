#pragma once

#include <QObject>
#include <PBDSolver.h>

class Scenario : public QObject
{
	Q_OBJECT

public:
	void initialize();
	void update(float deltaTime);
	void setPBDSolver(std::shared_ptr<PBDSolver> pbdSolver);

protected:
	virtual void onInitialize() = 0;
	virtual void onUpdate(float deltaTime) = 0;

	std::shared_ptr<PBDSolver> m_pbdSolver;
};

