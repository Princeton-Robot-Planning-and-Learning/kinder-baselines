(define (stream packing3d)
  (:stream sample-place-pose
    :inputs (?o)
    :domain (Packable ?o)
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Supported ?o ?p))
  )
  (:stream sample-grasp
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )
  (:stream inverse-kinematics
    :inputs (?o ?p ?g)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q ?t)
    :certified (and (BConf ?q) (ATraj ?t) (Kin ?o ?p ?g ?q ?t))
  )
  (:stream plan-base-motion
    :inputs (?q1 ?q2)
    :domain (and (BConf ?q1) (BConf ?q2))
    :fluents (AtPose)
    :outputs (?t)
    :certified (BaseMotion ?q1 ?t ?q2)
  )

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  )
  (:stream test-cfree-traj-pose
    :inputs (?t ?o2 ?p2)
    :domain (and (ATraj ?t) (Pose ?o2 ?p2))
    :certified (CFreeTrajPose ?t ?o2 ?p2)
  )
)
