#pragma once

#include "Scenario/Scenario.h"

class ObstacleCourseScenario : public Scenario
{
	Q_OBJECT

public:
	ObstacleCourseScenario();
	~ObstacleCourseScenario();

	virtual void onInitialize() override;
	virtual void onUpdate(float deltaTime) override;
	static const QString NAME;

private:
	float m_elapsedTime;
	QVector<std::shared_ptr<BoxConstraint>> m_dynamicObstacles;
};
