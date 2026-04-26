from __future__ import annotations



import flowpilot as fp





def main() -> None:

    engine = fp.FlowPilot()

    data = [

        {

            "sample_id": "s1",

            "events": [

                {"timestamp": 0, "entity_id": "u1", "features": {"value": 1.0}},

                {"timestamp": 1, "entity_id": "u1", "features": {"value": 1.2}},

            ],

            "label": 1,

        }

    ]



    result_from_function = fp.run(data=data, epochs=1, batch_size=1)

    result_from_object = engine.fit(data=data, epochs=1, batch_size=1)



    print("flowpilot import OK")

    print("version:", fp.__version__)

    print("function kpi time%:", result_from_function.kpi.training_time_reduction_pct)

    print("object kpi time%:", result_from_object.kpi.training_time_reduction_pct)





if __name__ == "__main__":

    main()

