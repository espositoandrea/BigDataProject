.TL
Oslo's Public Transport Dataset
.AU
Andrea Esposito
.AU
Graziano Montanaro
.AI
University of Bari \(lqAldo Moro\(rq
Department of Computer Science
Course: Big Data (6 CFU)
.NH 1
.XN Dataset Features
.LP
This section holds the documentation for any feature that are available in the original dataset. The explanation of each feature is purely derived from reasoning on their name and type conducted by the authors of this document.
.IP "\fBdateTime\fP" 0.4i
The date and time of the collection of the single data entry. Formatted following the ISO 8601 standard, minus the time zone (YYY-MMM-DD HH:MM:SS).
.IP "\fBLinkDistance\fP" 0.4i
An integer.
.I "We don't know what it means"
.IP "\fBPercentage\fP" 0.4i
A percentage (a floating point number with two decimal digits).
.I "We don't know what it means"
.IP "\fBLineRef\fP" 0.4i
A string representing the identifier of the public transport line.
.IP "\fBDirectionRef\fP" 0.4i
An integer.
.I "We don't know what it means"
.IP "\fBPublishedLineName\fP" 0.4i
A string representing the name of the public transport line. This is based on \(rqLineRef\(lq, but it is the name that is shown to the actual final user.
.IP "\fBOriginRef\fP" 0.4i
A string showing the reference to the departure station identifier.
.IP "\fBOriginName\fP" 0.4i
A string showing the common name of the departure station.
.I "Note:"
this value will often be modified by the data cleaning step as to fix various encoding errors that were introduced in the original EBCDIC to UTF8 conversion.
.IP "\fBDestinationRef\fP" 0.4i
A string showing the reference to the arrival station identifier.
.IP "\fBDestinationName\fP" 0.4i
A string showing the common name of the arrival station.
.I "Note:"
this value will often be modified by the data cleaning step as to fix various encoding errors that were introduced in the original EBCDIC to UTF8 conversion.
.IP "\fBOriginAimedDepartureTime\fP" 0.4i
An ISO 8601 timestamp holding the official departure time for the specified day.
.IP "\fBDestinationAimedArrivalTime\fP" 0.4i
An ISO 8601 timestamp holding the official arrival time for the specified day.
.IP "\fBVehicleRef\fP" 0.4i
An integer (though represented as float) holding the reference to the physical vehicle that will be used for the line in the specified date and time.
.IP "\fBDelay\fP" 0.4i
A string, supposedly showing the delay of the line.
.I "It uses a predefined format that we don't know or understand"
.IP "\fBHeadwayService\fP" 0.4i
A boolean.
.I Maybe
indicating if the line should be considered as not needing a timetable.
.IP "\fBInCongestion\fP" 0.4i
A boolean.
.I Maybe
showing if the line is stuck in traffic.
.IP "\fBInPanic\fP" 0.4i
A boolean.
.I Maybe
showing if the vehicle has had a failure and is inable to proceed with the line.
.IP "\fBLongitude\fP" 0.4i
The longitude of the bus at the specified timestamp.
.IP "\fBLatitude\fP" 0.4i
The latitude of the bus at the specified timestamp.
.IP "\fBmonitoredCall/StopPointRef\fP" 0.4i
A string indicating a reference to a stop station.
.I "We don't know what it means"
.IP "\fBmonitoredCall/VisitNumber\fP" 0.4i
An integer (though formatted as float).
.I "We don't know what it means"
.IP "\fBmonitoredCall/StopPointName\fP" 0.4i
A string indicating the common name for the stop station identified by \*QmonitoredCall/StopPointRef\*U
.I "We don't know what it means"
.IP "\fBmonitoredCall/VehicleAtStop\fP" 0.4i
A boolean.
.I "We don't know what it means" .
.I Maybe
indicates if the bus is stopped or not at the specified timestamp.
.IP "\fBmonitoredCall/DestinationDisplay\fP" 0.4i
A string.
.I "We don't know what it means" .
.I Maybe
what the bus display shows.
.NH 1
.XN Added Features
.LP
This section holds the documentation for any feature that was added during the initial clustering phase.
.IP "\fBCluster\fP" 0.4i
The identifier of the cluster to which the longitude and latitude of the bus belong.
.IP "\fBClusterLongitude\fP" 0.4i
The longitude of the cluster centroid.
.IP "\fBClusterLatitude\fP" 0.4i
The latitude of the cluster centroid.
.pdfsync
