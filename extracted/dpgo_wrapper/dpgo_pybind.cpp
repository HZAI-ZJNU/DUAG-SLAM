// extracted/dpgo_wrapper/dpgo_pybind.cpp
//
// Minimal pybind11 wrapper around DPGO's PGOAgent for distributed pose optimization.
// API verified against repos/dpgo/include/DPGO/PGOAgent.h

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <DPGO/PGOAgent.h>
#include <DPGO/DPGO_types.h>
#include <DPGO/manifold/Poses.h>

namespace py = pybind11;
using namespace DPGO;

PYBIND11_MODULE(dpgo_pybind, m) {
    m.doc() = "Minimal pybind11 bindings for DPGO distributed pose graph optimization";

    // PGOAgentParameters
    py::class_<PGOAgentParameters>(m, "PGOAgentParameters")
        .def(py::init([](unsigned d, unsigned r, unsigned numRobots) {
            return PGOAgentParameters(d, r, numRobots);
        }), py::arg("d") = 3, py::arg("r") = 3, py::arg("num_robots") = 1)
        .def_readwrite("d", &PGOAgentParameters::d)
        .def_readwrite("r", &PGOAgentParameters::r)
        .def_readwrite("numRobots", &PGOAgentParameters::numRobots)
        .def_readwrite("maxNumIters", &PGOAgentParameters::maxNumIters)
        .def_readwrite("verbose", &PGOAgentParameters::verbose);

    // RelativeSEMeasurement
    py::class_<RelativeSEMeasurement>(m, "RelativeSEMeasurement")
        .def(py::init<size_t, size_t, size_t, size_t,
                       const Eigen::MatrixXd &, const Eigen::VectorXd &,
                       double, double>(),
             py::arg("r1"), py::arg("r2"),
             py::arg("p1"), py::arg("p2"),
             py::arg("R"), py::arg("t"),
             py::arg("kappa"), py::arg("tau"))
        .def_readwrite("r1", &RelativeSEMeasurement::r1)
        .def_readwrite("r2", &RelativeSEMeasurement::r2)
        .def_readwrite("p1", &RelativeSEMeasurement::p1)
        .def_readwrite("p2", &RelativeSEMeasurement::p2)
        .def_readwrite("R", &RelativeSEMeasurement::R)
        .def_readwrite("t", &RelativeSEMeasurement::t)
        .def_readwrite("kappa", &RelativeSEMeasurement::kappa)
        .def_readwrite("tau", &RelativeSEMeasurement::tau);

    // PGOAgentStatus
    py::class_<PGOAgentStatus>(m, "PGOAgentStatus")
        .def(py::init<>())
        .def_readwrite("agentID", &PGOAgentStatus::agentID)
        .def_readwrite("iterationNumber", &PGOAgentStatus::iterationNumber)
        .def_readwrite("readyToTerminate", &PGOAgentStatus::readyToTerminate);

    // PGOAgent
    py::class_<PGOAgent>(m, "PGOAgent")
        .def(py::init<unsigned, const PGOAgentParameters &>(),
             py::arg("id"), py::arg("params"))
        .def("getID", &PGOAgent::getID)
        .def("num_poses", &PGOAgent::num_poses)
        .def("dimension", &PGOAgent::dimension)
        .def("iteration_number", &PGOAgent::iteration_number)

        // Measurement management
        .def("addMeasurement", &PGOAgent::addMeasurement, py::arg("measurement"))
        .def("setMeasurements", &PGOAgent::setMeasurements,
             py::arg("odometry"), py::arg("private_lc"), py::arg("shared_lc"))

        // Initialization
        .def("initialize", [](PGOAgent &agent) {
            agent.initialize();
        })

        // Optimization
        .def("iterate", &PGOAgent::iterate, py::arg("do_optimization") = true)

        // Neighbor management
        .def("setNeighborStatus", &PGOAgent::setNeighborStatus, py::arg("status"))
        .def("hasNeighbor", &PGOAgent::hasNeighbor, py::arg("neighbor_id"))
        .def("getNeighbors", &PGOAgent::getNeighbors)

        // Pose access
        .def("getPoseInGlobalFrame", [](PGOAgent &agent, unsigned pose_id) -> Eigen::MatrixXd {
            Eigen::MatrixXd T;
            bool ok = agent.getPoseInGlobalFrame(pose_id, T);
            if (!ok) {
                throw std::runtime_error("getPoseInGlobalFrame failed for pose " + std::to_string(pose_id));
            }
            return T;
        }, py::arg("pose_id"))

        .def("getTrajectoryInLocalFrame", [](PGOAgent &agent) -> Eigen::MatrixXd {
            Eigen::MatrixXd traj;
            bool ok = agent.getTrajectoryInLocalFrame(traj);
            if (!ok) {
                throw std::runtime_error("getTrajectoryInLocalFrame failed");
            }
            return traj;
        })

        .def("getTrajectoryInGlobalFrame", [](PGOAgent &agent) -> Eigen::MatrixXd {
            Eigen::MatrixXd traj;
            bool ok = agent.getTrajectoryInGlobalFrame(traj);
            if (!ok) {
                throw std::runtime_error("getTrajectoryInGlobalFrame failed");
            }
            return traj;
        })

        // Termination
        .def("shouldTerminate", &PGOAgent::shouldTerminate)

        // Reset
        .def("reset", &PGOAgent::reset);
}
