#pragma once

#include "Scenario/Scenario.h"

class MovingWallScenario : public Scenario
{
	Q_OBJECT

public:
	MovingWallScenario();
	~MovingWallScenario();

	virtual void onInitialize() override;
	virtual void onUpdate(float deltaTime) override;
	static const QString NAME;

private:
	float m_elapsedTime;
	std::shared_ptr<BoxConstraint> m_movingWall;
};
