// RUN: %clang --analyze %s -o %t > /dev/null 2>&1
// RUN: FileCheck -input-file %t %s

void test_null_init(void) {
  int *p = 0;
  *p = 0xDEADBEEF;
}

void test_null_assign(void) {
  int *p;
  p = 0;
  *p = 0xDEADBEEF;
}

void test_null_assign_transitive(void) {
  int *p;
  p = 0;
  int *q = p;
  *q = 0xDEADBEEF;
}

void test_null_cond(int *p) {
  if (!p) {
    *p = 0xDEADBEEF;
  }
}

void test_null_cond_transitive(int *q) {
  if (!q) {
    int *p = q;
    *p = 0xDEADBEEF;
  }
}

void test_null_field(void) {
  struct s { int *p; } x;
  x.p = 0;
  *(x.p) = 0xDEADBEEF;
}

void test_assumptions(int a, int b)
{
  if (a == 0) {
    return;
  }
  if (b != 0) {
    return;
  }
  int *p = 0;
  *p = 0xDEADBEEF;
}

// CHECK: <?xml version="1.0" encoding="UTF-8"?>
// CHECK: <!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
// CHECK: <plist version="1.0">
// CHECK: <dict>
// CHECK:  <key>files</key>
// CHECK:  <array>
// CHECK:  </array>
// CHECK:  <key>diagnostics</key>
// CHECK:  <array>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>5</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>5</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>6</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>6</integer>
// CHECK:            <key>col</key><integer>4</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>6</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>6</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>6</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:      <key>message</key>
// CHECK: <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Dereference of null pointer</string>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>6</integer>
// CHECK:    <key>col</key><integer>3</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>10</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>10</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>12</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>12</integer>
// CHECK:            <key>col</key><integer>4</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>12</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>12</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>12</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:      <key>message</key>
// CHECK: <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Dereference of null pointer</string>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>12</integer>
// CHECK:    <key>col</key><integer>3</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>16</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>16</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>19</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>19</integer>
// CHECK:            <key>col</key><integer>4</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>19</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>19</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>19</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Dereference of null pointer (loaded from variable &apos;q&apos;)</string>
// CHECK:      <key>message</key>
// CHECK: <string>Dereference of null pointer (loaded from variable &apos;q&apos;)</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Dereference of null pointer (loaded from variable &apos;q&apos;)</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Dereference of null pointer</string>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>19</integer>
// CHECK:    <key>col</key><integer>3</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>23</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>23</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>23</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>23</integer>
// CHECK:            <key>col</key><integer>8</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>23</integer>
// CHECK:       <key>col</key><integer>7</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>23</integer>
// CHECK:          <key>col</key><integer>7</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>23</integer>
// CHECK:          <key>col</key><integer>8</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Assuming &apos;p&apos; is null</string>
// CHECK:      <key>message</key>
// CHECK: <string>Assuming &apos;p&apos; is null</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>23</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>23</integer>
// CHECK:            <key>col</key><integer>8</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>24</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>24</integer>
// CHECK:            <key>col</key><integer>6</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>24</integer>
// CHECK:       <key>col</key><integer>5</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>24</integer>
// CHECK:          <key>col</key><integer>6</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>24</integer>
// CHECK:          <key>col</key><integer>6</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:      <key>message</key>
// CHECK: <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Dereference of null pointer</string>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>24</integer>
// CHECK:    <key>col</key><integer>5</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>29</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>29</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>29</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>29</integer>
// CHECK:            <key>col</key><integer>8</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>29</integer>
// CHECK:       <key>col</key><integer>7</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>29</integer>
// CHECK:          <key>col</key><integer>7</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>29</integer>
// CHECK:          <key>col</key><integer>8</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Assuming &apos;q&apos; is null</string>
// CHECK:      <key>message</key>
// CHECK: <string>Assuming &apos;q&apos; is null</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>29</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>29</integer>
// CHECK:            <key>col</key><integer>8</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>30</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>30</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>30</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>30</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>31</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>31</integer>
// CHECK:            <key>col</key><integer>6</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>31</integer>
// CHECK:       <key>col</key><integer>5</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>31</integer>
// CHECK:          <key>col</key><integer>6</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>31</integer>
// CHECK:          <key>col</key><integer>6</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:      <key>message</key>
// CHECK: <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Dereference of null pointer</string>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>31</integer>
// CHECK:    <key>col</key><integer>5</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>36</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>36</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>36</integer>
// CHECK:            <key>col</key><integer>10</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>36</integer>
// CHECK:            <key>col</key><integer>10</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>36</integer>
// CHECK:            <key>col</key><integer>10</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>36</integer>
// CHECK:            <key>col</key><integer>10</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>38</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>38</integer>
// CHECK:            <key>col</key><integer>8</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>38</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>38</integer>
// CHECK:          <key>col</key><integer>7</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>38</integer>
// CHECK:          <key>col</key><integer>7</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Dereference of null pointer (loaded from field &apos;p&apos;)</string>
// CHECK:      <key>message</key>
// CHECK: <string>Dereference of null pointer (loaded from field &apos;p&apos;)</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Dereference of null pointer (loaded from field &apos;p&apos;)</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Dereference of null pointer</string>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>38</integer>
// CHECK:    <key>col</key><integer>3</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>43</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>43</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>43</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>43</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>43</integer>
// CHECK:       <key>col</key><integer>7</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>43</integer>
// CHECK:          <key>col</key><integer>7</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>43</integer>
// CHECK:          <key>col</key><integer>12</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Assuming &apos;a&apos; is not equal to 0</string>
// CHECK:      <key>message</key>
// CHECK: <string>Assuming &apos;a&apos; is not equal to 0</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>43</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>43</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>46</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>46</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>46</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>46</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>46</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>46</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>46</integer>
// CHECK:       <key>col</key><integer>7</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>46</integer>
// CHECK:          <key>col</key><integer>7</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>46</integer>
// CHECK:          <key>col</key><integer>12</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Assuming &apos;b&apos; is equal to 0</string>
// CHECK:      <key>message</key>
// CHECK: <string>Assuming &apos;b&apos; is equal to 0</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>46</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>46</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>49</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>49</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>49</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>49</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>50</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>50</integer>
// CHECK:            <key>col</key><integer>4</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>50</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>50</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>50</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:      <key>message</key>
// CHECK: <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Dereference of null pointer</string>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>50</integer>
// CHECK:    <key>col</key><integer>3</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:  </array>
// CHECK: </dict>
// CHECK: </plist>
