#pragma once

#include "Scenario/Scenario.h"

class ShapeTypeComparisonScenario : public Scenario
{
	Q_OBJECT

public:
	ShapeTypeComparisonScenario();
	~ShapeTypeComparisonScenario();

	virtual void onInitialize() override;
	virtual void onUpdate(float deltaTime) override;
	static const QString NAME;
};
