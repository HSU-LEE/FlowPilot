from __future__ import annotations



import sys



import flowpilot as fp





def main() -> int:

    data = [

        {

            "sample_id": "g1",

            "events": [

                {"timestamp": 0, "entity_id": "u1", "features": {"value": 1.0}},

                {"timestamp": 1, "entity_id": "u1", "features": {"value": 1.4}},

            ],

            "label": 1,

        },

        {

            "sample_id": "g2",

            "events": [

                {"timestamp": 0, "entity_id": "u2", "features": {"value": 0.4}},

                {"timestamp": 1, "entity_id": "u2", "features": {"value": 0.7}},

            ],

            "label": 0,

        },

    ]



    harness = fp.BenchmarkHarness(runs=3, seed=42)

    pilot = fp.FlowPilot()

    results = harness.run(pilot=pilot, data=data, epochs=1, batch_size=1)



    failures = [item.baseline for item in results if not harness.release_gate_passed(item)]

    if failures:

        print("Release gate failed for baselines:", ", ".join(failures))

        return 1

    print("Release gate passed")

    return 0





if __name__ == "__main__":

    sys.exit(main())

